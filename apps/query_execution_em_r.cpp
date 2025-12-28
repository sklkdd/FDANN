// ====================================================================================
// query_execution_em_r.cpp - EM+R Query Support for FDANN
// ====================================================================================

#include <cstring>
#include <iomanip>
#include <algorithm>
#include <numeric>
#include <omp.h>
#include <set>
#include <string.h>
#include <boost/program_options.hpp>
#include <fstream>
#include <sstream>

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

// Global atomic to store peak thread count (must match extern declaration in global_thread_counter.h)
std::atomic<int> peak_threads(1);

namespace po = boost::program_options;

// ====================================================================================
// EM+R Data Structures
// ====================================================================================

struct EMRQueryAttribute {
    int em_value;
    int r_start;
    int r_end;
};

// Read database R values from EM+R file (format: <em>,<r> per line)
std::vector<int> read_database_r_values(const std::string& path) {
    std::vector<int> r_values;
    std::ifstream file(path);
    if (!file.is_open()) {
        std::cerr << "ERROR: Cannot open database attributes file: " << path << std::endl;
        return r_values;
    }
    std::string line;
    while (std::getline(file, line)) {
        size_t comma = line.find(',');
        if (comma != std::string::npos) {
            int r_val = std::stoi(line.substr(comma + 1));
            r_values.push_back(r_val);
        }
    }
    return r_values;
}

// Read query attributes from EM+R file (format: <em>,<r_start>-<r_end> per line)
std::vector<EMRQueryAttribute> read_em_r_query_attributes(const std::string& path) {
    std::vector<EMRQueryAttribute> queries;
    std::ifstream file(path);
    if (!file.is_open()) {
        std::cerr << "ERROR: Cannot open query attributes file: " << path << std::endl;
        return queries;
    }
    std::string line;
    while (std::getline(file, line)) {
        EMRQueryAttribute q;
        size_t comma = line.find(',');
        if (comma != std::string::npos) {
            q.em_value = std::stoi(line.substr(0, comma));
            std::string range_part = line.substr(comma + 1);
            size_t dash = range_part.find('-');
            if (dash != std::string::npos) {
                q.r_start = std::stoi(range_part.substr(0, dash));
                q.r_end = std::stoi(range_part.substr(dash + 1));
                queries.push_back(q);
            }
        }
    }
    return queries;
}

// Extract EM-only filters from EM+R query attributes
std::vector<std::string> extract_em_only_filters(const std::vector<EMRQueryAttribute>& em_r_queries) {
    std::vector<std::string> em_filters;
    for (const auto& q : em_r_queries) {
        em_filters.push_back(std::to_string(q.em_value));
    }
    return em_filters;
}

// Post-filter results by range and compute recall
double compute_em_r_recall(
    const std::vector<uint32_t>& result_ids,  // K results per query, flattened
    const std::vector<std::vector<uint32_t>>& gt_ids,  // Ground truth per query
    const std::vector<EMRQueryAttribute>& em_r_queries,
    const std::vector<int>& r_values,
    size_t query_num,
    size_t k,
    size_t expanded_k
) {
    double total_recall = 0.0;
    size_t valid_queries = 0;
    
    for (size_t q = 0; q < query_num; q++) {
        const EMRQueryAttribute& query = em_r_queries[q];
        
        // Post-filter results by range constraint
        std::vector<uint32_t> filtered_results;
        for (size_t i = 0; i < expanded_k && filtered_results.size() < k; i++) {
            uint32_t idx = result_ids[q * expanded_k + i];
            if (idx < r_values.size()) {
                int r_val = r_values[idx];
                if (r_val >= query.r_start && r_val <= query.r_end) {
                    filtered_results.push_back(idx);
                }
            }
        }
        
        // Compute recall for this query
        if (q < gt_ids.size() && gt_ids[q].size() > 0) {
            std::set<uint32_t> gt_set(gt_ids[q].begin(), gt_ids[q].begin() + std::min(k, gt_ids[q].size()));
            std::set<uint32_t> result_set(filtered_results.begin(), filtered_results.end());
            
            size_t intersection = 0;
            for (uint32_t id : result_set) {
                if (gt_set.count(id) > 0) {
                    intersection++;
                }
            }
            
            double recall = (double)intersection / (double)gt_set.size();
            total_recall += recall;
            valid_queries++;
        }
    }
    
    return valid_queries > 0 ? total_recall / valid_queries : 0.0;
}

template <typename T, typename LabelT = uint32_t>
int search_memory_index_em_r(
    diskann::Metric &metric, 
    const std::string &index_path,
    const std::string &query_file, 
    const std::string &truthset_file, 
    const std::string &em_r_db_attrs_file,
    const std::string &em_r_query_attrs_file,
    const uint32_t num_threads,
    const uint32_t recall_at,
    const std::vector<uint32_t> &Lvec,
    const uint32_t expansion_factor  // How many more candidates to request for post-filtering
) {
    // Restrict number of threads to 1 for query execution
    omp_set_num_threads(1);

    // Monitor thread count
    std::atomic<bool> done(false);
    std::thread monitor(monitor_thread_count, std::ref(done));

    using TagT = uint32_t;
    
    // Load ground truth from .ivecs file (same approach as query_execution.cpp)
    // We use read_ivecs() from fanns_survey_helpers.cpp rather than diskann::load_truthset()
    // because our ground truth is in .ivecs format, not .bin format
    std::vector<std::vector<int>> ground_truth_raw = read_ivecs(truthset_file);
    if (ground_truth_raw.empty()) {
        std::cerr << "ERROR: Failed to load ground truth from " << truthset_file << std::endl;
        return -1;
    }
    size_t gt_num = ground_truth_raw.size();
    size_t gt_dim = ground_truth_raw[0].size();
    
    // Convert ground truth to vector<vector<uint32_t>> format
    std::vector<std::vector<uint32_t>> gt_ids(gt_num);
    for (size_t i = 0; i < gt_num; i++) {
        gt_ids[i].reserve(gt_dim);
        for (size_t j = 0; j < ground_truth_raw[i].size(); j++) {
            gt_ids[i].push_back(static_cast<uint32_t>(ground_truth_raw[i][j]));
        }
    }
    std::cout << "Loaded ground truth: " << gt_num << " queries, " << gt_dim << " neighbors each" << std::endl;

    // Load query vectors
    T *query_data = nullptr;
    size_t query_num, query_dim, query_aligned_dim;
    diskann::load_aligned_bin<T>(query_file, query_data, query_num, query_dim, query_aligned_dim);

    // Load EM+R database attributes (for R values)
    std::vector<int> r_values = read_database_r_values(em_r_db_attrs_file);
    if (r_values.empty()) {
        std::cerr << "ERROR: Failed to load database R values" << std::endl;
        return -1;
    }
    std::cout << "Loaded " << r_values.size() << " database R values" << std::endl;

    // Load EM+R query attributes
    std::vector<EMRQueryAttribute> em_r_queries = read_em_r_query_attributes(em_r_query_attrs_file);
    if (em_r_queries.empty()) {
        std::cerr << "ERROR: Failed to load EM+R query attributes" << std::endl;
        return -1;
    }
    // Truncate to match query_num
    if (em_r_queries.size() > query_num) {
        em_r_queries.resize(query_num);
    }
    std::cout << "Loaded " << em_r_queries.size() << " EM+R queries" << std::endl;

    // Extract EM-only filters for FDANN
    std::vector<std::string> em_only_filters = extract_em_only_filters(em_r_queries);

    // Load FDANN index
    std::cout << "Loading index from " << index_path << std::endl;
    bool filtered_search = true;
    auto config = diskann::IndexConfigBuilder()
                      .with_metric(metric)
                      .with_dimension(query_dim)
                      .with_max_points(0)
                      .with_data_load_store_strategy(diskann::DataStoreStrategy::MEMORY)
                      .with_graph_load_store_strategy(diskann::GraphStoreStrategy::MEMORY)
                      .with_data_type(diskann_type_to_name<T>())
                      .with_label_type(diskann_type_to_name<LabelT>())
                      .with_tag_type(diskann_type_to_name<TagT>())
                      .is_dynamic_index(false)
                      .is_enable_tags(false)
                      .is_filtered(filtered_search)
                      .is_pq_dist_build(false)
                      .with_num_pq_chunks(0)
                      .build();

    auto index_factory = diskann::IndexFactory(config);
    auto index = index_factory.create_instance();
    index->load(index_path.c_str(), num_threads, 100);
    std::cout << "Index loaded" << std::endl;

    // Expanded K for retrieving more candidates (use size_t for safety on large datasets)
    size_t expanded_k = static_cast<size_t>(recall_at) * static_cast<size_t>(expansion_factor);
    
    // Memory safety check: ensure we don't try to allocate excessive memory
    // Each result array needs: expanded_k * query_num * sizeof(uint32_t or float) = expanded_k * query_num * 4 bytes
    // With two arrays, total = expanded_k * query_num * 8 bytes
    size_t total_results = expanded_k * query_num;
    size_t memory_required_bytes = total_results * (sizeof(uint32_t) + sizeof(float));
    const size_t MAX_MEMORY_BYTES = 8ULL * 1024 * 1024 * 1024;  // 8 GB limit
    
    if (memory_required_bytes > MAX_MEMORY_BYTES) {
        std::cerr << "ERROR: Memory requirement (" << memory_required_bytes / (1024*1024) 
                  << " MB) exceeds limit (" << MAX_MEMORY_BYTES / (1024*1024) << " MB)." << std::endl;
        std::cerr << "Consider reducing expansion_factor (current: " << expansion_factor 
                  << ") or using fewer queries." << std::endl;
        return -1;
    }
    
    std::cout << "Memory allocation: " << memory_required_bytes / (1024*1024) << " MB for result arrays" << std::endl;
    
    std::cout << "\n=== EM+R Query Execution ===" << std::endl;
    std::cout << "K = " << recall_at << ", Expanded K = " << expanded_k << ", Expansion Factor = " << expansion_factor << std::endl;
    std::cout << std::setw(4) << "L" << std::setw(12) << "QPS" << std::setw(20) << "EM+R Recall@" << recall_at << std::endl;
    std::cout << "====================================================" << std::endl;

    double best_recall = 0.0;
    double qps_fanns_survey = -1.0;
    
    // Pre-allocate result storage ONCE (outside the loop for efficiency)
    std::vector<uint32_t> query_result_ids;
    std::vector<float> query_result_dists;
    try {
        query_result_ids.resize(total_results);
        query_result_dists.resize(total_results);
    } catch (const std::bad_alloc& e) {
        std::cerr << "ERROR: Failed to allocate memory for results: " << e.what() << std::endl;
        std::cerr << "Required: " << memory_required_bytes / (1024*1024) << " MB" << std::endl;
        return -1;
    }

    for (uint32_t L : Lvec) {
        // FDANN requires L >= K in search_with_filters (index.cpp line 2051)
        // Since we pass expanded_k as K, we must ensure L >= expanded_k
        uint32_t effective_L = std::max(L, static_cast<uint32_t>(expanded_k));
        
        auto start = std::chrono::high_resolution_clock::now();
        
        // Execute filtered search for each query
        for (size_t i = 0; i < query_num; i++) {
            index->search_with_filters(
                query_data + i * query_aligned_dim,
                em_only_filters[i],  // EM-only filter
                expanded_k,          // Request more candidates
                effective_L,         // Search list size (must be >= expanded_k)
                query_result_ids.data() + i * expanded_k,
                query_result_dists.data() + i * expanded_k
            );
        }
        
        auto end = std::chrono::high_resolution_clock::now();
        double elapsed_seconds = std::chrono::duration<double>(end - start).count();
        double qps = query_num / elapsed_seconds;
        
        // Store QPS for output
        if (qps_fanns_survey < 0.0) {
            qps_fanns_survey = qps;
        }
        
        // Compute EM+R recall with post-filtering
        double em_r_recall = compute_em_r_recall(
            query_result_ids, gt_ids, em_r_queries, r_values,
            query_num, recall_at, expanded_k
        );
        
        std::cout << std::setw(4) << L << std::setw(12) << qps << std::setw(20) << em_r_recall << std::endl;
        best_recall = std::max(em_r_recall, best_recall);
    }

    // Stop thread monitor
    done = true;
    monitor.join();

    // Output statistics in the format expected by fdann.py
    // (same format as query_execution.cpp for parser compatibility)
    printf("Maximum number of threads: %d\n", peak_threads.load()-1);  // Subtract 1 for monitoring thread
    peak_memory_footprint();  // Outputs VmPeak and VmHWM lines
    printf("Queries per second: %.3f\n", qps_fanns_survey);
    printf("Recall: %.3f\n", best_recall);

    // Cleanup
    delete[] query_data;
    // Note: gt_ids is a vector, automatically cleaned up

    return 0;
}

int main(int argc, char **argv) {
    std::string data_type, dist_fn, index_path_prefix, query_file, gt_file;
    std::string em_r_db_attrs_file, em_r_query_attrs_file;
    uint32_t num_threads, K, expansion_factor;
    std::vector<uint32_t> Lvec;

    po::options_description desc{"Arguments"};

    try {
        desc.add_options()("help,h", "Print information on arguments");

        po::options_description required_configs("Required");
        required_configs.add_options()
            ("data_type", po::value<std::string>(&data_type)->required(), "data type (float)")
            ("dist_fn", po::value<std::string>(&dist_fn)->required(), "distance function (l2)")
            ("index_path_prefix", po::value<std::string>(&index_path_prefix)->required(), "Path prefix to the index")
            ("query_file", po::value<std::string>(&query_file)->required(), "Query file in bin format")
            ("gt_file", po::value<std::string>(&gt_file)->required(), "EM+R ground truth file")
            ("em_r_db_attrs", po::value<std::string>(&em_r_db_attrs_file)->required(), "EM+R database attributes file")
            ("em_r_query_attrs", po::value<std::string>(&em_r_query_attrs_file)->required(), "EM+R query attributes file")
            ("K,K", po::value<uint32_t>(&K)->required(), "Number of neighbors to return")
            ("L,L", po::value<std::vector<uint32_t>>(&Lvec)->multitoken()->required(), "Search list size")
            ("expansion_factor", po::value<uint32_t>(&expansion_factor)->default_value(10), 
             "Expansion factor for candidate retrieval (default: 10)");

        po::options_description optional_configs("Optional");
        optional_configs.add_options()
            ("num_threads,T", po::value<uint32_t>(&num_threads)->default_value(1), "Number of threads");

        desc.add(required_configs).add(optional_configs);

        po::variables_map vm;
        po::store(po::parse_command_line(argc, argv, desc), vm);

        if (vm.count("help")) {
            std::cout << desc << std::endl;
            return 0;
        }

        po::notify(vm);
    } catch (const std::exception &ex) {
        std::cerr << ex.what() << std::endl;
        return -1;
    }

    diskann::Metric metric;
    if (dist_fn == "l2") {
        metric = diskann::Metric::L2;
    } else if (dist_fn == "mips") {
        metric = diskann::Metric::INNER_PRODUCT;
    } else if (dist_fn == "cosine") {
        metric = diskann::Metric::COSINE;
    } else {
        std::cerr << "Unknown distance function: " << dist_fn << std::endl;
        return -1;
    }

    if (data_type == "float") {
        return search_memory_index_em_r<float>(
            metric, index_path_prefix, query_file, gt_file,
            em_r_db_attrs_file, em_r_query_attrs_file,
            num_threads, K, Lvec, expansion_factor
        );
    } else {
        std::cerr << "Unsupported data type: " << data_type << std::endl;
        return -1;
    }
}
