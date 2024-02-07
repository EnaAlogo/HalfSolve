#pragma once

#include <optional>
#include <assert.h>


namespace smath {

    struct slice_t {
        constexpr slice_t(
            std::optional<int64_t> start,   
            std::optional<int64_t> stop , 
            std::optional<int64_t> step = std::nullopt )
            :m_start(start), m_stop(stop), m_step(step.value_or(1)) {
            if (m_step == 0) { throw std::runtime_error("step cannot be zero"); };
        }
        constexpr explicit slice_t(std::optional<int64_t> stop)
            :slice_t{ std::nullopt , stop , std::nullopt } {};

        constexpr int64_t len() const {
            return std::max(0ll, (get_stop() - get_start() + m_step + (m_step > 0 ? -1 : 1)) / m_step);
        }
        constexpr int64_t len(int64_t dim) const {
            return std::max(0ll, (get_stop(dim) - get_start(dim) + m_step + (m_step > 0 ? -1 : 1)) / m_step);
        }
        constexpr std::optional<int64_t> start() const { return m_start; };
        constexpr std::optional<int64_t> stop() const { return m_stop; };
        constexpr int64_t step() const { return m_step; };
        constexpr int64_t get_start() const { return m_start.value_or(0ll); };
        constexpr int64_t get_stop() const { assert(m_stop.has_value()); return m_stop.value(); };
        constexpr int64_t get_stop(int64_t dim) const {
            if (m_stop) {
                if (int Stop = m_stop.value(); Stop < 0) {
                    return std::max(-1ll, Stop + dim);
                }
                return std::min(m_stop.value(), dim);
            }
            return m_step > 0 ? dim : -1;
        };
        constexpr int64_t get_start(int64_t dim) const {
            if (m_start) {
                int64_t start = m_step > 0 ? 0 : -1;
                if (int Start = m_start.value(); Start < 0) {
                    return std::max(start, Start + dim);
                }
                return std::min(m_start.value(), dim + start);
            }
            return m_step > 0 ? 0 : dim - 1;
        }
    private:
        std::optional<int64_t> m_start  , m_stop;
        int64_t m_step = 1;

    };


    struct range_t {
        class range_iterator {
        public:
            using iterator_category = std::bidirectional_iterator_tag;
            using difference_type = std::ptrdiff_t;
            using value_type = int64_t;
            using pointer = int64_t*;
            using reference = int64_t&;
            using const_reference = int64_t const&;
            using const_pointer = int64_t const*;
            using self = range_iterator;
            constexpr range_iterator()
                :index(0), step(1) {};
            constexpr range_iterator(int64_t start, int64_t step)
                :index(start), step(step) {};
            constexpr range_iterator(const  range_iterator& other)
                :index(other.index), step(other.step) {};
            constexpr range_iterator& operator=(const  range_iterator& other) = default;

            constexpr range_iterator operator++(int) {
                self copy = *this;
                index += step;
                return copy;
            };
            constexpr range_iterator& operator++() {
                index += step;
                return *this;
            };
            constexpr range_iterator operator--(int) {
                self copy = *this;
                index -= step;
                return copy;
            }
            constexpr range_iterator& operator--() {
                index -= step;
                return *this;
            }

            constexpr self& operator+=(difference_type rhs) { index += rhs; return *this; } 
            constexpr self& operator-=(difference_type rhs) { index -= rhs; return *this; } 
            constexpr difference_type operator-(const self& rhs) const { return index - rhs.index; } 
            constexpr self operator+(difference_type rhs) const { return self(index + rhs,step); }
            constexpr self operator-(difference_type rhs) const { return self(index - rhs,step); }
            friend constexpr self operator+(difference_type lhs, const self& rhs) { return self(lhs + rhs.index, rhs.step); }
            friend constexpr self operator-(difference_type lhs, const self& rhs) {  return self(lhs - rhs.index, rhs.step);}

            constexpr value_type operator*() const { return index; }; 
            constexpr bool operator ==(const range_iterator& other) const { return step > 0 ? index >= other.index : index <= other.index; };
            constexpr bool operator !=(const range_iterator& other) const { return !(*this == other); };
            constexpr bool operator>(const range_iterator& rhs) const{ return  index >  rhs.index; }
            constexpr bool operator<(const range_iterator& rhs) const { return  index <  rhs.index; }
            constexpr bool operator>=(const range_iterator& rhs)const { return index >= rhs.index; }
            constexpr bool operator<=(const range_iterator& rhs) const { return index <= rhs.index; }
        private:
            /*
            * TODO: 
            * will changing this to contain the slice and 
            * turning it into a random_access iterator make a 
            * difference for the better?
            */
            int64_t index, step;
        };

        using iterator = range_iterator;
        using const_iterator = range_iterator;


        constexpr range_t(int64_t stop)
            :slice(stop) {};
        constexpr range_t(int64_t start, int64_t stop, int64_t step = 1)
            :slice(start, stop, step) {};

        constexpr const_iterator begin() const {
            return const_iterator{ slice.get_start() , slice.step() };
        }
        constexpr const_iterator end() const {
            return slice.len() == 0 ? const_iterator{ slice.get_start() , slice.step() }
            : const_iterator{ slice.get_stop() , slice.step() };
        }



    private:
        slice_t slice;
    };


};