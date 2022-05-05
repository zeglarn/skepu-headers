namespace skepu
{
	template <typename T>
	Index1D VectorIterator<T>::getIndex() const
	{
		size_t index = *this - m_parent.globalBegin();
		return Index1D{index};
	}
	
	template <typename T>
	VectorIterator<T>::VectorIterator(parent_type& parent, T *std_iterator) : m_parent(parent), m_std_iterator(std_iterator) {}
	
	template <typename T>
	typename VectorIterator<T>::parent_type& VectorIterator<T>::getParent() const
	{
		return m_parent;
	}
	
	template <typename T>
	VectorIterator<T>& VectorIterator<T>::begin()
	{
		return *this;
	}
	
	// Returns begin() if s >= 0, else returns end() - 1
	template <typename T>
	typename Vector<T>::strided_iterator VectorIterator<T>::stridedBegin(size_t n, int s)
	{
		return typename Vector<T>::strided_iterator(this->m_parent, &this->m_std_iterator[(s < 0) ? (-n + 1) * s : 0], s);
	}
	
	// Returns the number of elements remaining, starting from the iterator
	template <typename T>
	size_t VectorIterator<T>::size()
	{
		return this->m_parent.end() - *this;
	}
	
	template <typename T>
	T* VectorIterator<T>::getAddress() const
	{
		return m_std_iterator;
	}
	
	template <typename T>
	T* VectorIterator<T>::data()
	{
		return m_std_iterator;
	}
	
	template <typename T>
	T& VectorIterator<T>::operator()(const ssize_t index)
	{
#ifdef SKEPU_MPI
		m_parent.mark_dirty();
#endif
		return m_std_iterator[index];
	}
	
	template <typename T>
	const T& VectorIterator<T>::operator()(const ssize_t index) const 
	{
		return m_std_iterator[index];
	}

	template <typename T>
	T& VectorIterator<T>::operator[](const ssize_t index)
	{
		m_parent.updateHost();
		m_parent.invalidateDeviceData();
#ifdef SKEPU_MPI
		m_parent.mark_dirty();
#endif
		return m_std_iterator[index];
	}
	
	template <typename T>
	const T& VectorIterator<T>::operator[](const ssize_t index) const
	{
		m_parent.updateHost();
		return m_std_iterator[index];
	}
	
	template <typename T>
	bool VectorIterator<T>::operator==(const iterator& i)
	{
		return (m_std_iterator == i.m_std_iterator);
	}
	
	template <typename T>
	bool VectorIterator<T>::operator!=(const iterator& i)
	{
		return (m_std_iterator != i.m_std_iterator);
	}
	
	template <typename T>
	bool VectorIterator<T>::operator<(const iterator& i)
	{
		return (m_std_iterator < i.m_std_iterator);
	}
	
	template <typename T>
	bool VectorIterator<T>::operator>(const iterator& i)
	{
		return (m_std_iterator > i.m_std_iterator);
	}
	
	template <typename T>
	bool VectorIterator<T>::operator<=(const iterator& i)
	{
		return (m_std_iterator <= i.m_std_iterator);
	}
	
	template <typename T>
	bool VectorIterator<T>::operator>=(const iterator& i)
	{
		return (m_std_iterator >= i.m_std_iterator);
	}
	
	template <typename T>
	const VectorIterator<T>& VectorIterator<T>::operator++() //Prefix
	{
		++m_std_iterator;
		return *this;
	}
	
	template <typename T>
	VectorIterator<T> VectorIterator<T>::operator++(int) //Postfix
	{
		iterator temp(*this);
		++m_std_iterator;
		return temp;
	}
	
	template <typename T>
	const VectorIterator<T>& VectorIterator<T>::operator--() //Prefix
	{
		--m_std_iterator;
		return *this;
	}
	
	template <typename T>
	VectorIterator<T> VectorIterator<T>::operator--(int) //Postfix
	{
		iterator temp(*this);
		--m_std_iterator;
		return temp;
	}
	
	template <typename T>
	const VectorIterator<T>& VectorIterator<T>::operator+=(const ssize_t i)
	{
		m_std_iterator += i;
		return *this;
	}
	
	template <typename T>
	const VectorIterator<T>& VectorIterator<T>::operator-=(const ssize_t i)
	{
		m_std_iterator -= i;
		return *this;
	}
	
	template <typename T>
	VectorIterator<T> VectorIterator<T>::operator-(const ssize_t i) const
	{
		iterator temp(*this);
		temp -= i;
		return temp;
	}
	
	template <typename T>
	VectorIterator<T> VectorIterator<T>::operator+(const ssize_t i) const
	{
		iterator temp(*this);
		temp += i;
		return temp;
	}
	
	template <typename T>
	typename VectorIterator<T>::parent_type::difference_type VectorIterator<T>::operator-(const iterator& i) const
	{
		return m_std_iterator - i.m_std_iterator;
	}
	
	template <typename T>
	T& VectorIterator<T>::operator*()
	{
		m_parent.updateHost();
		m_parent.invalidateDeviceData();
		return *m_std_iterator;
	}
	
	template <typename T>
	const T& VectorIterator<T>::operator*() const
	{
		m_parent.updateHost();
		return *m_std_iterator;
	}
	
	template <typename T>
	const T& VectorIterator<T>::operator-> () const
	{
		m_parent.updateHost();
		return *m_std_iterator;
	}
	
	template <typename T>
	T& VectorIterator<T>::operator-> ()
	{
		m_parent.updateHost();
		m_parent.invalidateDeviceData();
		return *m_std_iterator;
	}
	
}
