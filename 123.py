from __future__ import annotations

from typing import List, MutableSequence, Sequence, TypeVar

T = TypeVar("T")


def _median_of_three_index(a: MutableSequence[T], lo: int, hi: int) -> int:
	mid = lo + (hi - lo) // 2
	x, y, z = a[lo], a[mid], a[hi]
	if x < y:
		if y < z:
			return mid
		return hi if x < z else lo
	else:
		if x < z:
			return lo
		return hi if y < z else mid


def _partition_lomuto(a: MutableSequence[T], lo: int, hi: int) -> int:
	pivot = a[hi]
	i = lo - 1
	for j in range(lo, hi):
		if a[j] <= pivot:
			i += 1
			a[i], a[j] = a[j], a[i]
	p = i + 1
	a[p], a[hi] = a[hi], a[p]
	return p


def _quick_sort_inplace(a: MutableSequence[T], lo: int, hi: int) -> None:
	if lo >= hi:
		return
	pivot_index = _median_of_three_index(a, lo, hi)
	a[pivot_index], a[hi] = a[hi], a[pivot_index]

	p = _partition_lomuto(a, lo, hi)

	_quick_sort_inplace(a, lo, p - 1)
	_quick_sort_inplace(a, p + 1, hi)


def quick_sort_inplace(a: MutableSequence[T]) -> MutableSequence[T]:
	if len(a) <= 1:
		return a
	_quick_sort_inplace(a, 0, len(a) - 1)
	return a


def quick_sort(a: Sequence[T]) -> List[T]:
	copied = list(a)
	quick_sort_inplace(copied)
	return copied


def _self_test() -> None:
	cases = [
		[],
		[1],
		[2, 1],
		[3, 1, 2],
		[1, 2, 3, 4, 5],
		[5, 4, 3, 2, 1],
		[3, 3, 3, 2, 2, 1],
		[9, -1, 0, 5, 2, 2, 7, 3],
	]

	for arr in cases:
		expected = sorted(arr)
		got = quick_sort(arr)
		assert got == expected, (arr, got, expected)

		arr2 = list(arr)
		quick_sort_inplace(arr2)
		assert arr2 == expected, (arr, arr2, expected)


if __name__ == "__main__":
	_self_test()

	data = [9, 1, 8, 3, 7, 2, 6, 5, 4]
	print("原始:", data)
	quick_sort_inplace(data)
	print("排序:", data)