#include <algorithm>
#include <numeric>
#include <ad3/GenericFactor.h>

namespace AD3 {

class _Permutahedron : public GenericFactor {

public:

    // standard boilerplate
    _Permutahedron() {}
    virtual ~_Permutahedron() { ClearActiveSet(); }


    void Initialize(int length)
    {
        //SetVerbosity(100);
        length_ = length;
        cached_permutation_.resize(length_);
        std::iota(std::begin(cached_permutation_),
                  std::end(cached_permutation_),
                  0);
    }

    void Maximize(const vector<double>& scores,
                  const vector<double>&,
                  Configuration& configuration,
                  double* value)
    override
    {
        // argsort scores, warm-starting from cached permutation
        std::sort(
            std::begin(cached_permutation_),
            std::end(cached_permutation_),
            [&scores](int i, int j)
            {
                return scores[i] < scores[j];
            }
        );

        // compute objective value <inv(perm), scores>
        // simultaneously, invert permutation: configuration = inv(cached_permutation_)
        *value = 0;
        auto* perm = static_cast<std::vector<int>*>(configuration);
        for (int i = 0; i < length_; ++i)
        {
            *value += (i + 1) * scores[cached_permutation_[i]];
            (*perm)[cached_permutation_[i]] = i + 1;
        }

        // function returns `value` and `perm` via output arguments.
    }

    void Evaluate(const vector<double>& scores,
                  const vector<double>&,
                  const Configuration configuration,
                  double* value)
    override
    {
        auto* perm = static_cast<const std::vector<int>*>(configuration);

        // compute objective value <perm, scores>
        *value = 0;
        for (int i = 0; i < length_; ++i)
        {
            auto ix = static_cast<double>((*perm)[i]);
            *value += ix * scores[i];
        }
    }

    void UpdateMarginalsFromConfiguration(const Configuration& configuration,
                                          double weight,
                                          vector<double>* marg_u,
                                          vector<double>*)
    override
    {
        auto* perm = static_cast<const std::vector<int>*>(configuration);
        for (int i = 0; i < length_; ++i)
        {
            auto ix = static_cast<double>((*perm)[i]);
            (*marg_u)[i] += weight * ix;
        }
    }

    // For this polytope, this function is a misnomer.
    // We really want the dot product between the vertices.
    int CountCommonValues(const Configuration& configuration1,
                          const Configuration& configuration2)
    override
    {
        auto* perm1 = static_cast<const std::vector<int>*>(configuration1);
        auto* perm2 = static_cast<const std::vector<int>*>(configuration2);

        assert(perm1->size() == length_);
        assert(perm2->size() == length_);

        int dpfast = 0;
        for (int i = 0; i < length_; ++i)
        {
            dpfast += (*perm1)[i] * (*perm2)[i];
        }
        return dpfast;
    }

    // Check if two configurations are the same.
    bool SameConfiguration(const Configuration& configuration1,
                           const Configuration& configuration2)
    override
    {
        auto* perm1 = static_cast<const std::vector<int>*>(configuration1);
        auto* perm2 = static_cast<const std::vector<int>*>(configuration2);

        assert(perm1->size() == length_);
        assert(perm2->size() == length_);

        for (int i = 0; i < length_; ++i) {
            if ((*perm1)[i] != (*perm2)[i])
                return false;
        }
        return true;
    }

    // boilerplate required by AD3
    Configuration CreateConfiguration()
    override
    {
        std::vector<int>* sequence = new std::vector<int>(length_, -1);
        return static_cast<Configuration>(sequence);
    }

    void DeleteConfiguration(Configuration configuration)
    override
    {
        std::vector<int>* sequence = static_cast<std::vector<int>*>(configuration);
        delete sequence;
    }

    virtual size_t GetNumAdditionals() override { return num_additionals_; }

protected:

    int length_;
    int num_additionals_ = 0;
    std::vector<int> cached_permutation_;

}; // class Permutahedron

} // namespace
