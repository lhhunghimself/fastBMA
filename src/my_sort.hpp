//sort routines -modified original C routines by Ariel Faigon, 1987 that used a paired key value structure and macros
//replaced with cpp class and templates
template <class T> class sort_key_t{
	public:
  T value;
  int key;
};	
template <class T> void  partial_quickersort (sort_key_t <T> *array, int lower, int upper);
template <class T> void  sedgesort (sort_key_t <T>  *array, int len);
template <class T> void  insort (sort_key_t <T> *array, int len);
template <class T> void sort_by_scores (int nsize, T *scores, int *index, bool min_first_flag);

template<class T> void sort_by_scores (int nsize, T *scores, int *index, bool min_first_flag){
 T max=std::numeric_limits<T>::max();
 sort_key_t <T> *vk=new sort_key_t<T>[nsize+1];
 vk[nsize].value=max; 
 vk[nsize].key=nsize;
 for (int i=0;i<nsize;i++){
  vk[i].value=scores[i];
  vk[i].key=i;
 }
 sedgesort(vk,nsize+1);
 if(!min_first_flag){
	 for(int i=0;i<nsize;i++)
		 index[i]=vk[nsize-i-1].key;
	}	
	else{
  for(int i=0;i<nsize;i++)
   index[i]=vk[i].key;
 }
 delete [] vk;
}

template <class T> void  sedgesort (sort_key_t <T>  *array, int len){
   partial_quickersort (array, 0, len - 1);
   insort (array, len);
}

template <class T> void  insort (sort_key_t <T> *array, int len){
	sort_key_t <T> temp;
	for (int i = 1; i < len; i++) {
		/* invariant:  array[0..i-1] is sorted */
		int j = i;
		/* customization bug: SWAP is not used here */
	 temp = array[j];
		while (j > 0 && (array[j-1].value > temp.value)) {
			array[j] = array[j-1];
			j--;
		}
		array[j] = temp;
	}
}
template <class T> void  partial_quickersort (sort_key_t <T> *array, int lower, int upper){
	//careful here if using unsigned type upper < lower breaks since you can't have negative number
 int	i, j;
 sort_key_t <T>	temp, pivot;
 if (upper - lower > 50) {
		temp = array[lower]; array[lower] = array[(upper+lower)/2]; array[(upper+lower)/2] = temp;
	 i = lower;  j = upper + 1;  pivot = array[lower];
	 while (1){
	  do i++; while (array[i].value<pivot.value);
	  do j--; while (array[j].value>pivot.value);
	  if (j < i) break;
	  temp =array[i]; array[i]=array[j]; array[j]=temp;
	 }
	 temp=array[lower]; array[lower]=array[j];array[j]=temp;
	 partial_quickersort (array, lower, j - 1);
	 partial_quickersort (array, i, upper);
 }
}
