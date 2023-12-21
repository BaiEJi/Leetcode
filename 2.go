package main

type TreeNode struct {
	Val   int
	Left  *TreeNode
	Right *TreeNode
}

func matrixScore(grid [][]int) int {
	// 首列
	for i := 0; i < len(grid); i++ {
		if grid[i][0] == 0 {
			for j := 0; j < len(grid[i]); j++ {
				//grid[i][j] = 1 - grid[i][j]
				grid[i][j] ^= 1
			}
		}
	}

	// 对于之后的每一列
	for j := 1; j < len(grid[0]); j++ {
		cnt := 0 // 记录本列的0的个数
		for i := 0; i < len(grid); i++ {
			if grid[i][j] == 0 {
				cnt++
			}
		}

		if cnt > len(grid)/2 {
			for i := 0; i < len(grid); i++ {
				grid[i][j] ^= 1
			}
		}
	}

	res := 0
	for i := 0; i < len(grid); i++ {
		tmpRes := 0
		for j := 0; j < len(grid[i]); j++ {
			tmpRes = (tmpRes << 1) + grid[i][j]
		}
		res += tmpRes
	}
	return res

}
