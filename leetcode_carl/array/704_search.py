# 704 search
# debug part
nums = [-1,0,3,5,9,12]
target = 9

len_nums = len(nums)
left = 0 
right = len_nums

while left < right:
    mid = (left + right) // 2
    if nums[mid] > target:
        right = mid 
    elif nums[mid] < target:
        left = mid + 1
    else:
        print(mid)
        break

print (-1)





# leetcode actual part
# class Solution:
#     def search(self, nums: List[int], target: int) -> int:
#         len_nums = len(nums)
#         left = 0 
#         right = len_nums

#         while left < right:
#             mid = (left + right) // 2
#             if nums[mid] > target:
#                 right = mid 
#             elif nums[mid] < target:
#                 left = mid + 1
#             else:
#                 return mid

#         return -1
        




