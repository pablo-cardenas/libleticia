#include <limits>
#include <memory>
#include <omp.h>
#include <queue>
#include <stdio.h>

using namespace std;

vector<int>
neighbors(int u, int n_row, int n_col)
{
    vector<int> ret;

    int u_row = u / n_col;
    int u_col = u % n_row;

    for (auto v_row : { u_row + 1, u_row - 1 }) {
        if (0 <= v_row && v_row < n_row) {
            ret.push_back(v_row * n_col + u_col);
        }
    }

    for (auto v_col : { u_col + 1, u_col - 1 }) {
        if (0 <= v_col && v_col < n_col) {
            ret.push_back(n_col * u_row + v_col);
        }
    }

    return ret;
}

void
dijkstra_line(const char* grid,
              size_t n_row,
              size_t n_col,
              size_t u0,
              double taken_weight,
              double* dist)
{
    priority_queue<pair<float, int>,
                   vector<pair<float, int>>,
                   greater<pair<float, int>>>
      pq;

    for (size_t u = 0; u < n_row * n_col; ++u) {
        dist[u] = std::numeric_limits<double>::infinity();
    }

    for (size_t j = 0; j < n_col; ++j) {
        pq.push({ 0, 0 * n_col + j });
        dist[0 * n_col + j] = 0;
    }

    while (!pq.empty()) {
        size_t u = pq.top().second;
        pq.pop();

        double weight = 1;
        if (grid[u] || u == u0) {
            weight = taken_weight;
        }

        for (size_t v : neighbors(u, n_row, n_col)) {

            if (dist[v] > dist[u] + weight) {
                dist[v] = dist[u] + weight;
                pq.push({ dist[v], v });
            }
        }
    }
}

extern "C" void
taken_distances(const char* grid,
                size_t n_row,
                size_t n_col,
                double taken_weight,
                double* output,
                double* base)
{

    // omp_set_num_threads(4);
    double time = omp_get_wtime();

    // Compute taken vector
    vector<size_t> taken;
    for (size_t u = 0; u < n_row * n_col; ++u) {
        if (grid[u]) {
            taken.push_back(u);
        }
    }

#pragma omp parallel for
    for (size_t u0 = 0; u0 < n_row * n_col; ++u0) {
        double* dist = new double[n_row * n_col];
        dijkstra_line(grid, n_row, n_col, u0, taken_weight, dist);

        for (size_t k = 0; k < taken.size(); ++k) {
            output[u0 * taken.size() + k] = dist[taken[k]];
        }
    }

    double* dist = new double[n_row * n_col];
    dijkstra_line(grid, n_row, n_col, -1, taken_weight, dist);
    for (size_t k = 0; k < taken.size(); ++k) {
        base[k] = dist[taken[k]];
    }

    time = omp_get_wtime() - time;
    // printf("taken_distances: %fs\n", time);
}

extern "C" void
distances(const char* grid,
          size_t n_row,
          size_t n_col,
          double taken_weight,
          double* output)
{
    dijkstra_line(grid, n_row, n_col, -1, taken_weight, output);
}

void
dijkstra(const char* grid,
         size_t n_row,
         size_t n_col,
         size_t u0,
         size_t source,
         double taken_weight,
         double* dist)
{
    priority_queue<pair<float, int>,
                   vector<pair<float, int>>,
                   greater<pair<float, int>>>
      pq;

    for (size_t u = 0; u < n_row * n_col; ++u) {
        dist[u] = std::numeric_limits<double>::infinity();
    }

    pq.push({ 0, source });
    dist[source] = 0;

    // First iteration without taken_weight
    size_t u = pq.top().second;
    pq.pop();
    double weight = 1;
    for (size_t v : neighbors(u, n_row, n_col)) {

        if (dist[v] > dist[u] + weight) {
            dist[v] = dist[u] + weight;
            pq.push({ dist[v], v });
        }
    }

    // following iterations
    while (!pq.empty()) {
        size_t u = pq.top().second;
        pq.pop();

        double weight = 1;
        if (grid[u] || u == u0) {
            weight = taken_weight;
        }

        for (size_t v : neighbors(u, n_row, n_col)) {

            if (dist[v] > dist[u] + weight) {
                dist[v] = dist[u] + weight;
                pq.push({ dist[v], v });
            }
        }
    }
}

extern "C" void
neighbors_distances(const char* grid,
                    size_t n_row,
                    size_t n_col,
                    double taken_weight,
                    double* output,
                    double* distances)
{
    // omp_set_num_threads(4);
    double time = omp_get_wtime();

    // Compute taken vector
    vector<size_t> taken;
    for (size_t u = 0; u < n_row * n_col; ++u) {
        if (grid[u]) {
            taken.push_back(u);
        }
    }

#pragma omp parallel for
    for (size_t u0 = 0; u0 < n_row * n_col; ++u0) {
        double* dist = new double[n_row * n_col];
        for (size_t k = 0; k < taken.size(); ++k) {
            dijkstra(grid, n_row, n_col, u0, taken[k], taken_weight, dist);

            for (size_t h = 0; h < taken.size(); ++h) {
                output[(u0 * taken.size() + k) * taken.size() + h] =
                  dist[taken[h]];
            }
        }
    }

    double* dist = new double[n_row * n_col];
    for (size_t k = 0; k < taken.size(); ++k) {
        dijkstra(grid, n_row, n_col, -1, taken[k], taken_weight, dist);
        for (size_t h = 0; h < taken.size(); ++h) {
            distances[k * taken.size() + h] = dist[taken[h]];
        }
    }

    time = omp_get_wtime() - time;
    // printf("neighbors_distances: %fs\n", time);
}
