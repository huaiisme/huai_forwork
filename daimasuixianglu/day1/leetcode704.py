class Solution:
    def search(self, nums: List[int], target: int) -> int:
        len_nums = len(nums)
        left = 0 
        right = len_nums
        while left < right:
            middle = (left + right) // 2
            if nums[middle] > target:
                right = middle
            elif nums[middle] < target:
                left = middle + 1 
            else:
                return middle
        
        return -1