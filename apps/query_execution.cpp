// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT license.

#include <cstring>
#include <iomanip>
#include <algorithm>
#include <numeric>
#include <omp.h>
#include <set>
#include <string.h>
#include <boost/program_options.hpp>

#ifndef _WINDOWS
#include <sys/mman.h>
#include <sys/stat.h>
#include <time.h>
#include <unistd.h>
#endif

#include "index.h"
#include "memory_mapper.h"
#include "utils.h"
#include "program_options_utils.hpp"
#include "index_factory.h"


#include <atomic>
#include "fanns_survey_helpers.cpp"
#include "global_thread_counter.h"

// Global atomic to store peak thread count
std::atomic<int> peak_threads(1);

namespace po = boost::program_options;

template <typename T, typename LabelT = uint32_t>
int search_memory_index(diskann::Metric &metric, const std::string &index_path, const std::string &result_path_prefix,
                        const std::string &query_file, const std::string &truthset_file, const uint32_t num_threads,
                        const uint32_t recall_at, const bool print_all_recalls, const std::vector<uint32_t> &Lvec,
                        const bool dynamic, const bool tags, const bool show_qps_per_thread,
                        const std::vector<std::string> &query_filters, const float fail_if_recall_below)
{
    // Restrict number of threads to 1 for query execution
    omp_set_num_threads(1);

    // Monitor thread count
    std::atomic<bool> done(false);
    std::thread monitor(monitor_thread_count, std::ref(done));

    using TagT = uint32_t;
    // Load the query file
    T *query = nullptr;
    uint32_t *gt_ids = nullptr;
    float *gt_dists = nullptr;
    size_t query_num, query_dim, query_aligned_dim, gt_num, gt_dim;
    diskann::load_aligned_bin<T>(query_file, query, query_num, query_dim, query_aligned_dim);

    bool calc_recall_flag = false;
	// We modified this to load the ground truth as .ivecs rather than .bin
	// We do not load the ground-truth-distances. The recall computation function should still work.
	// We do not need to truncate the ground-truth to K entries here, this is done in the calculate_recall function.
    if (truthset_file != std::string("null") && file_exists(truthset_file))
    {
		// Read ground truth from .ivecs file
		std::vector<std::vector<int>> ground_truth = read_ivecs(truthset_file);
		// Extract info
		gt_num = ground_truth.size();
		gt_dim = ground_truth[0].size();
		// Check if the number of queries matches the ground truth
        if (gt_num != query_num)
        {
            std::cout << "Error. Mismatch in number of queries and ground truth data" << std::endl;
        }
		// Allocate space for ground truth ids
		gt_ids = new uint32_t[gt_num * gt_dim];
		// Flatten the 2D vector into the 1D array
		for (size_t i = 0; i < gt_num; ++i) {
			for (size_t j = 0; j < gt_dim; ++j) {
				gt_ids[i * gt_dim + j] = static_cast<uint32_t>(ground_truth[i][j]);
			}
		}
        calc_recall_flag = true;
    }
    else
    {
        diskann::cout << " Truthset file " << truthset_file << " not found. Not computing recall." << std::endl;
    }

    bool filtered_search = false;
    if (!query_filters.empty())
    {
        filtered_search = true;
        if (query_filters.size() != 1 && query_filters.size() != query_num)
        {
            std::cout << "Error. Mismatch in number of queries and size of query "
                         "filters file"
                      << std::endl;
            return -1; // To return -1 or some other error handling?
        }
    }

    const size_t num_frozen_pts = diskann::get_graph_num_frozen_points(index_path);

    auto config = diskann::IndexConfigBuilder()
                      .with_metric(metric)
                      .with_dimension(query_dim)
                      .with_max_points(0)
                      .with_data_load_store_strategy(diskann::DataStoreStrategy::MEMORY)
                      .with_graph_load_store_strategy(diskann::GraphStoreStrategy::MEMORY)
                      .with_data_type(diskann_type_to_name<T>())
                      .with_label_type(diskann_type_to_name<LabelT>())
                      .with_tag_type(diskann_type_to_name<TagT>())
                      .is_dynamic_index(dynamic)
                      .is_enable_tags(tags)
                      .is_concurrent_consolidate(false)
                      .is_pq_dist_build(false)
                      .is_use_opq(false)
                      .with_num_pq_chunks(0)
                      .with_num_frozen_pts(num_frozen_pts)
                      .build();

    auto index_factory = diskann::IndexFactory(config);
    auto index = index_factory.create_instance();
    index->load(index_path.c_str(), num_threads, *(std::max_element(Lvec.begin(), Lvec.end())));
    std::cout << "Index loaded" << std::endl;

    if (metric == diskann::FAST_L2)
        index->optimize_index_layout();

    std::cout << "Using " << num_threads << " threads to search" << std::endl;
    std::cout.setf(std::ios_base::fixed, std::ios_base::floatfield);
    std::cout.precision(2);
    const std::string qps_title = show_qps_per_thread ? "QPS/thread" : "QPS";
    uint32_t table_width = 0;
    if (tags)
    {
        std::cout << std::setw(4) << "Ls" << std::setw(12) << qps_title << std::setw(20) << "Mean Latency (mus)"
                  << std::setw(15) << "99.9 Latency";
        table_width += 4 + 12 + 20 + 15;
    }
    else
    {
        std::cout << std::setw(4) << "Ls" << std::setw(12) << qps_title << std::setw(18) << "Avg dist cmps"
                  << std::setw(20) << "Mean Latency (mus)" << std::setw(15) << "99.9 Latency";
        table_width += 4 + 12 + 18 + 20 + 15;
    }
    uint32_t recalls_to_print = 0;
    const uint32_t first_recall = print_all_recalls ? 1 : recall_at;
    if (calc_recall_flag)
    {
        for (uint32_t curr_recall = first_recall; curr_recall <= recall_at; curr_recall++)
        {
            std::cout << std::setw(12) << ("Recall@" + std::to_string(curr_recall));
        }
        recalls_to_print = recall_at + 1 - first_recall;
        table_width += recalls_to_print * 12;
    }
    std::cout << std::endl;
    std::cout << std::string(table_width, '=') << std::endl;

    std::vector<std::vector<uint32_t>> query_result_ids(Lvec.size());
    std::vector<std::vector<float>> query_result_dists(Lvec.size());
    std::vector<float> latency_stats(query_num, 0);
    std::vector<uint32_t> cmp_stats;
    if (not tags || filtered_search)
    {
        cmp_stats = std::vector<uint32_t>(query_num, 0);
    }

    std::vector<TagT> query_result_tags;
    if (tags)
    {
        query_result_tags.resize(recall_at * query_num);
    }

    double best_recall = 0.0;

	double recall_fanns_survey = -1.0;
	double qps_fanns_survey = -1.0;

	// Lvec a vector of L values (search parameter). Four our framework, we only use one L value at a time.
    for (uint32_t test_id = 0; test_id < Lvec.size(); test_id++)
    {
        uint32_t L = Lvec[test_id];
        if (L < recall_at)
        {
            diskann::cout << "Ignoring search with L:" << L << " since it's smaller than K:" << recall_at << std::endl;
            continue;
        }

        query_result_ids[test_id].resize(recall_at * query_num);
        query_result_dists[test_id].resize(recall_at * query_num);
        std::vector<T *> res = std::vector<T *>();

        auto s = std::chrono::high_resolution_clock::now();
        omp_set_num_threads(num_threads);
#pragma omp parallel for schedule(dynamic, 1)
        for (int64_t i = 0; i < (int64_t)query_num; i++)
        {
            auto qs = std::chrono::high_resolution_clock::now();
            if (filtered_search && !tags)
            {
                std::string raw_filter = query_filters.size() == 1 ? query_filters[0] : query_filters[i];

                auto retval = index->search_with_filters(query + i * query_aligned_dim, raw_filter, recall_at, L,
                                                         query_result_ids[test_id].data() + i * recall_at,
                                                         query_result_dists[test_id].data() + i * recall_at);
                cmp_stats[i] = retval.second;
            }
            else if (metric == diskann::FAST_L2)
            {
                index->search_with_optimized_layout(query + i * query_aligned_dim, recall_at, L,
                                                    query_result_ids[test_id].data() + i * recall_at);
            }
            else if (tags)
            {
                if (!filtered_search)
                {
                    index->search_with_tags(query + i * query_aligned_dim, recall_at, L,
                                            query_result_tags.data() + i * recall_at, nullptr, res);
                }
                else
                {
                    std::string raw_filter = query_filters.size() == 1 ? query_filters[0] : query_filters[i];

                    index->search_with_tags(query + i * query_aligned_dim, recall_at, L,
                                            query_result_tags.data() + i * recall_at, nullptr, res, true, raw_filter);
                }

                for (int64_t r = 0; r < (int64_t)recall_at; r++)
                {
                    query_result_ids[test_id][recall_at * i + r] = query_result_tags[recall_at * i + r];
                }
            }
            else
            {
                cmp_stats[i] = index
                                   ->search(query + i * query_aligned_dim, recall_at, L,
                                            query_result_ids[test_id].data() + i * recall_at)
                                   .second;
            }
            auto qe = std::chrono::high_resolution_clock::now();
            std::chrono::duration<double> diff = qe - qs;
            latency_stats[i] = (float)(diff.count() * 1000000);
        }
        std::chrono::duration<double> diff = std::chrono::high_resolution_clock::now() - s;

        double displayed_qps = query_num / diff.count();

		// We use general QPS not QPS per thread
		// This assertion checks that we only use on L-value.
		assert(qps_fanns_survey < 0.0);
		qps_fanns_survey = displayed_qps;

        if (show_qps_per_thread)
            displayed_qps /= num_threads;

        std::vector<double> recalls;
        if (calc_recall_flag)
        {
            recalls.reserve(recalls_to_print);
            for (uint32_t curr_recall = first_recall; curr_recall <= recall_at; curr_recall++)
            {
                recalls.push_back(diskann::calculate_recall((uint32_t)query_num, gt_ids, gt_dists, (uint32_t)gt_dim,
                                                            query_result_ids[test_id].data(), recall_at, curr_recall));
            }
        }

        std::sort(latency_stats.begin(), latency_stats.end());
        double mean_latency =
            std::accumulate(latency_stats.begin(), latency_stats.end(), 0.0) / static_cast<float>(query_num);

        float avg_cmps = (float)std::accumulate(cmp_stats.begin(), cmp_stats.end(), 0) / (float)query_num;

        if (tags && !filtered_search)
        {
            std::cout << std::setw(4) << L << std::setw(12) << displayed_qps << std::setw(20) << (float)mean_latency
                      << std::setw(15) << (float)latency_stats[(uint64_t)(0.999 * query_num)];
        }
        else
        {
            std::cout << std::setw(4) << L << std::setw(12) << displayed_qps << std::setw(18) << avg_cmps
                      << std::setw(20) << (float)mean_latency << std::setw(15)
                      << (float)latency_stats[(uint64_t)(0.999 * query_num)];
        }
        for (double recall : recalls)
        {
            std::cout << std::setw(12) << recall;
            best_recall = std::max(recall, best_recall);
			// This assertion checks that we only compute one recall (recall@k) and we only use on L-value.
			assert(recall_fanns_survey < 0.0);
			recall_fanns_survey = recall / 100.0; // FDANN gives recall in percentage, we convert it to a number between 0 and 1
        }
        std::cout << std::endl;
    }

    // Stop thread count monitoring
    done = true;
    monitor.join();

	// Print results
	printf("Maximum number of threads: %d\n", peak_threads.load()-1);   // Subtract 1 because of the monitoring thread
	peak_memory_footprint();
	printf("Queries per second: %.3f\n", qps_fanns_survey);
	printf("Recall: %.3f\n", recall_fanns_survey);

	// Clean up
    diskann::aligned_free(query);
	delete[] gt_ids;

	// Return	
    return best_recall >= fail_if_recall_below ? 0 : -1;
}

int main(int argc, char **argv)
{
	// Parameters used in interface, have been there in original code
    std::string data_type;
	std::string dist_fn;
	std::string index_path_prefix;
	std::string result_path;
	std::string query_file;
	std::string query_filters_file;
	std::string gt_file;
	uint32_t K;
    std::vector<uint32_t> Lvec;

	// Hard-coded parameters
    uint32_t num_threads = 1;	// Use one thread for query execution
    bool print_all_recalls = false;
	bool show_qps_per_thread = false;
    float fail_if_recall_below = 0.0f;
	bool dynamic = false;
	bool tags = false;

    po::options_description desc{program_options_utils::make_program_description("search_memory_index", "Searches in-memory DiskANN indexes")};
    try
    {
        desc.add_options()("help,h", "Print this information on arguments");

        // Required parameters
        po::options_description required_configs("Required");
        required_configs.add_options()("data_type", po::value<std::string>(&data_type)->required(),
                                       program_options_utils::DATA_TYPE_DESCRIPTION);
        required_configs.add_options()("dist_fn", po::value<std::string>(&dist_fn)->required(),
                                       program_options_utils::DISTANCE_FUNCTION_DESCRIPTION);
        required_configs.add_options()("gt_file", po::value<std::string>(&gt_file)->default_value(std::string("null")),
                                       program_options_utils::GROUND_TRUTH_FILE_DESCRIPTION);
        required_configs.add_options()("index_path_prefix", po::value<std::string>(&index_path_prefix)->required(),
                                       program_options_utils::INDEX_PATH_PREFIX_DESCRIPTION);
        required_configs.add_options()("result_path", po::value<std::string>(&result_path)->required(),
                                       program_options_utils::RESULT_PATH_DESCRIPTION);
        required_configs.add_options()("query_file", po::value<std::string>(&query_file)->required(),
                                       program_options_utils::QUERY_FILE_DESCRIPTION);
        required_configs.add_options()("recall_at,K", po::value<uint32_t>(&K)->required(),
                                       program_options_utils::NUMBER_OF_RESULTS_DESCRIPTION);
        required_configs.add_options()("search_list,L",
                                       po::value<std::vector<uint32_t>>(&Lvec)->multitoken()->required(),
                                       program_options_utils::SEARCH_LIST_DESCRIPTION);
        required_configs.add_options()("query_filters_file",
                                       po::value<std::string>(&query_filters_file)->required(),
                                       program_options_utils::FILTERS_FILE_DESCRIPTION);
        desc.add(required_configs);

        po::variables_map vm;
        po::store(po::parse_command_line(argc, argv, desc), vm);
        if (vm.count("help"))
        {
            std::cout << desc;
            return 0;
        }
        po::notify(vm);
    }
    catch (const std::exception &ex)
    {
        std::cerr << ex.what() << '\n';
        return -1;
    }

	// Set correct distance metric
    diskann::Metric metric;
    if ((dist_fn == std::string("mips")) && (data_type == std::string("float")))
    {
        metric = diskann::Metric::INNER_PRODUCT;
    }
    else if (dist_fn == std::string("l2"))
    {
        metric = diskann::Metric::L2;
    }
    else if (dist_fn == std::string("cosine"))
    {
        metric = diskann::Metric::COSINE;
    }
    else if ((dist_fn == std::string("fast_l2")) && (data_type == std::string("float")))
    {
        metric = diskann::Metric::FAST_L2;
    }
    else
    {
        std::cout << "Unsupported distance function. Currently only l2/ cosine are "
                     "supported in general, and mips/fast_l2 only for floating "
                     "point data."
                  << std::endl;
        return -1;
    }

	// Read the query attributes
    std::vector<std::string> query_filters;
	if (query_filters_file != "")
    {
        query_filters = read_file_to_vector_of_strings(query_filters_file);
    }

	// Perform the search: We hard-code for float data
    try
    {
		return search_memory_index<float>(metric, index_path_prefix, result_path, query_file, gt_file,
											  num_threads, K, print_all_recalls, Lvec, dynamic, tags,
											  show_qps_per_thread, query_filters, fail_if_recall_below);
    }
    catch (std::exception &e)
    {
        std::cout << std::string(e.what()) << std::endl;
        diskann::cerr << "Index search failed." << std::endl;
        return -1;
    }
}
