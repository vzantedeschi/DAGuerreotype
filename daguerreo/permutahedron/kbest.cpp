#include <vector>
#include <iostream>
#include <limits>
#include <utility>

#include <torch/extension.h>

namespace py = pybind11;

using torch::indexing::Slice;
using torch::indexing::None;


at::Tensor invert_permutation(at::Tensor p) {
    auto res = at::zeros_like(p);
    auto inc = at::arange(p.size(0), at::kLong);
    res.index_put_({ p }, inc);
    return res;
}


class SortedData {
public:
    SortedData(at::Tensor x_)
        : perm_{ x_.argsort() }
        , inv_perm_ { invert_permutation(perm_) }
        , vals_{ x_.index({ perm_ }) }
        , diff_{ vals_.diff() }
        , perm{ perm_.accessor<int64_t, 1>() }
        , vals{ vals_.accessor<double, 1>() }
        , diff{ diff_.accessor<double, 1>() }
    { }

    at::Tensor perm_;
    at::Tensor inv_perm_;
    at::Tensor vals_;
    at::Tensor diff_;
    at::TensorAccessor<int64_t, 1> perm;
    at::TensorAccessor<double, 1> vals;
    at::TensorAccessor<double, 1> diff;
};


class State {
public:
    State(int64_t d, int64_t k_max)
        : topk_perms_ { at::zeros({ k_max, d }, at::kLong) }
        , topk_scores_ { at::zeros({ k_max }, at::kDouble) }
        , trans_flips_ { at::zeros({ k_max, d - 1 }, at::kLong) }
        , trans_scores_ { at::zeros({ k_max, d - 1 }, at::kDouble) }
        , topk_perms { topk_perms_.accessor<int64_t, 2>() }
        , topk_scores { topk_scores_.accessor<double, 1>() }
        , trans_flips { trans_flips_.accessor<int64_t, 2>() }
        , trans_scores { trans_scores_.accessor<double, 2>() }
        , j(k_max)
    { }

    at::Tensor topk_perms_;
    at::Tensor topk_scores_;
    at::Tensor trans_flips_;
    at::Tensor trans_scores_;
    at::TensorAccessor<int64_t, 2> topk_perms;
    at::TensorAccessor<double, 1> topk_scores;
    at::TensorAccessor<int64_t, 2> trans_flips;
    at::TensorAccessor<double, 2> trans_scores;
    std::vector<int64_t> j;
};


class KBestRankings {
public:
    KBestRankings(torch::Tensor x_, int64_t k_max)
        : d{ x_.size(0) }
        , k_max{ k_max }
        , k{ }
        , x{ x_ }
        , state{ d, k_max }
        , exhausted { false }
    {
        for (auto i = 0; i < d; ++i)
        {
           state.topk_perms[0][i] = i;
           state.topk_scores[0] += (1+i) * x.vals[i];
        }

        process_new_perm();

    }

    std::tuple<at::Tensor, at::Tensor>
    compute(int64_t kk) {

        while (k < kk && k < k_max && !exhausted) {
            next();
        }

        auto slice = Slice(None, k);
        auto sc = state.topk_scores_.index({ slice });
        auto tk = state.topk_perms_.index({ slice, x.inv_perm_ });
        return { tk, sc };
    }

    void process_new_perm() {

        auto p_k = state.topk_perms_.index({k});
        auto delta = torch::diff(p_k) * x.diff_;
        auto tra_ix = delta.argsort();
        auto tra_sc = -delta.index({ tra_ix }) + state.topk_scores[k];

        state.trans_flips_.index_put_({ k }, tra_ix);
        state.trans_scores_.index_put_({ k }, tra_sc);

        ++k;
    }

    bool is_seen(int64_t i) {

        // check whether the permutation we are just about to add
        // has already been seen

        // the tricky part is we avoid materializing the flip.

        // candidate flip
        auto jj = state.trans_flips[i][state.j[i]];

        for (auto ii = 0; ii < k; ++ii) {
            // we compare the new permutation against the ii'th one.

            bool found = true;
            for (auto j = 0; j < d; ++j) {

                auto tgt_j = j;
                if (j == jj)
                    tgt_j = jj + 1;
                else if (j == jj + 1)
                    tgt_j = jj;

                // break at first mismatch
                if (state.topk_perms[ii][j] != state.topk_perms[i][tgt_j]) {
                    found = false;
                    break;
                }
            }
            if (found)
                return true;
        }

        // return false if we broke (mismatched) for every perm
        return false;
    }

    void next() {

        if (k_max > 0 and k >= k_max) {
            return;
        }

        // Find the best candidate among the k queues
        auto best_score = -std::numeric_limits<double>::infinity();
        int64_t best_i = -1;
        for (auto i = 0; i < k; ++i) {

            // skip until the first permutation not-yet-seen.
            // note: each queue has length d-1

            // optimization: could quickly discard solution with score
            // strictly greater than the bound. But numerically risky.
            auto& ji = state.j[i];
            while (ji < d - 1 && is_seen(i)) {
                ++ji;
            }
            if (ji >= d - 1)
                continue;
            auto sc_i = state.trans_scores[i][ji];
            if (sc_i > best_score) {
                best_score = sc_i;
                best_i = i;
            }
        }

        if (best_i == -1) {
            // exhausted every possible perm
            exhausted = true;
            return;
        }

        // populate the correct permutation and score on position k

        auto p_i = state.topk_perms_.index({ best_i });
        state.topk_scores[k] = best_score;
        state.topk_perms_.index_put_({ k }, p_i);
        auto jj = state.trans_flips[best_i][state.j[best_i]];
        std::swap(state.topk_perms[k][jj], state.topk_perms[k][jj+1]);

        process_new_perm();
    }

    int64_t d;
    int64_t k_max;
    int64_t k;
    SortedData x;
    State state;
    bool exhausted;
};


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    py::class_<KBestRankings>(m, "KBestRankings")
        .def(py::init<torch::Tensor, int64_t>())
        .def("compute", &KBestRankings::compute)
        ;

#ifdef VERSION_INFO
    m.attr("__version__") = VERSION_INFO;
#else
    m.attr("__version__") = "dev";
#endif

}
