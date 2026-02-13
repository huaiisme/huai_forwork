import random
class Solution:
    def quicksort(self, nums, low, high):
        flag = nums[random.randint(low, high)]
        i, j = low, high

        while i < j:
            while nums[i] < flag:
                i += 1
            while nums[j] > flag:
                j -= 1
            if i <= j:
                nums[i], nums[j] = nums[j], nums[i]
                i += 1
                j -= 1
        if i < high:
            self.quicksort(nums, i , high)
        if j > low:
            self.quicksort(nums, low, j)

    def findKthLargest(self, nums: List[int], k: int) -> int:
        len_nums = len(nums)
        low = 0
        high = len_nums - 1

        self.quicksort(nums, low, high)
        nums.reverse()
        return nums[k-1]
        
     


        
