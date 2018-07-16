def twoSum(nums, target):
    l = [[index1, index2] for index1, x in enumerate(nums)\
                            for index2, y in enumerate(nums)\
                            if index1 != index2 and x != y and x + y == target]
    if l != None:
        return l
    else :
        return None

def twoSum2(nums, target):
    result = []
    for index, num in enumerate(nums):
        if index == len(nums) - 1:
            return result
        for index2 in range(index + 1, len(nums)):
            if num != nums[index2] and num + nums[index2] == target:
                result.append([index, index2])
    return result

def twoSum3(nums, target):
    for index, num in enumerate(nums):
        if(nums.__contains__(target - num)):
            index2 = nums.index(target - num)
            if(index2 != index):
                return [index, index2]
    return []

def twoSum4(nums, target):
    dict = {}
    for index, num in enumerate(nums):
        if (target - num) in dict:
            return [index, dict.get(target - num)]
        else :
            dict[num] = index

def removeDuplicate(nums):
    preIndex = 0
    for index in range(1, len(nums)):
        if nums[preIndex] != nums[index]:
            preIndex += 1
            nums[preIndex] = nums[index]

    return preIndex + 1

def removeDuplicate2(nums):
    pos = 0
    this_num = None
    for num in nums:
        if num != this_num:
            nums[pos] = num
            this_num = num
            pos += 1
    return pos

def maxProfit(prices):
    if (len(prices) == 0) or (len(prices) == 1):
        return 0

    result = 0
    pre = prices[result]
    for index in range(1, len(prices)):
        if prices[index - 1] > prices[index]:
            result += prices[index - 1] - pre
            pre = prices[index]

    if pre < prices[len(prices) - 1]:
        result += prices[len(prices) - 1] - pre

    return result

def maxProfit2(prices):
    profit = 0
    for index in range(1, len(prices)):
        if prices[index] > prices[index - 1]:
            profit += prices[index] - prices[index - 1]
    return profit


# testing
# nums = [7,1,5,3,6,4]
nums = [1,2,3,4,5]
# nums = [5,5,4,3,2,1]
# result = maxProfit2(nums)
# print(result)

for i in range(11, 5, -1):
    print(i)