package main

import (
	"bufio"
	"fmt"
	"os"
)

type node struct {
	// 线段树的节点信息
	// l, r 节点区间
	// value 当前的统计值
	// tag 懒标记
	l, r, value, tag int
}

type Tree struct {
	arr []*node
}

func NewTree(n int) *Tree {
	// n 节点个数, 开4n空间
	return &Tree{
		arr: make([]*node, 4*n),
	}
}

func (t *Tree) pushUp(k int) {
	// 向上更新
	t.arr[k].value = t.arr[k*2].value + t.arr[k*2+1].value
}

func (t *Tree) pushDown(k int) {
	// 向下更新
	if t.arr[k].tag != 0 {
		t.arr[k*2].value += t.arr[k].tag * (t.arr[k*2].r - t.arr[k*2].l + 1)
		t.arr[k*2+1].value += t.arr[k].tag * (t.arr[k*2+1].r - t.arr[k*2+1].l + 1)
		t.arr[k*2].tag += t.arr[k].tag // 现在是儿子们欠孙子们账了
		t.arr[k*2+1].tag += t.arr[k].tag
		t.arr[k].tag = 0 // 父亲的账还完了
	}
}

func (t *Tree) build(k, l, r int, nums []int) {
	// idx真实存储下标, l,r 区间端点
	// nums 原始数组
	t.arr[k] = &node{
		l:   l,
		r:   r,
		tag: 0,
	}
	// 叶子结束递归, 存储最值
	if l == r {
		t.arr[k].value = nums[l-1]
		return
	}
	// 不是叶子 递归
	mid := (l + r) / 2
	t.build(2*k, l, mid, nums)
	t.build(2*k+1, mid+1, r, nums)
	t.pushUp(k) // 左右孩子的值交给父亲
}

func (t *Tree) update(k, l, r, v int) {
	// 使用tag进行区间修改
	// k 真是存储的下标
	// l, r 区间
	// v 修改的值
	if l <= t.arr[k].l && r >= t.arr[k].r {
		// 此节点北区间完全覆盖
		t.arr[k].value += (t.arr[k].r - t.arr[k].l + 1) * v // 欠每个儿子一个v
		t.arr[k].tag += v                                   // 打上标记
		return
	}

	mid := (t.arr[k].l + t.arr[k].r) / 2
	// 必须向儿子们分账
	t.pushDown(k)
	if l <= mid {
		t.update(k*2, l, r, v)
	}
	if r > mid {
		t.update(k*2+1, l, r, v)
	}

	// 再向上 左右儿子必须都找完才会向上
	t.pushUp(k) // 他爹存着他们的value, 所以必须向上
}

func (t *Tree) query(k, l, r int) int {
	if t.arr[k].l >= l && t.arr[k].r <= r {
		// 区间覆盖法
		return t.arr[k].value
	}

	// 开始收缩
	mid := (t.arr[k].l + t.arr[k].r) / 2
	// 向下
	t.pushDown(k)
	// 临时变量
	tmpSum := 0
	if l <= mid {
		// 在左区间去找
		tmpSum += t.query(k*2, l, r)
	}
	if r > mid {
		// 在右区间去找
		tmpSum += t.query(k*2+1, l, r)
	}
	return tmpSum
}

func main() {
	var n, m int
	in := bufio.NewReader(os.Stdin)
	out := bufio.NewWriter(os.Stdout)
	defer out.Flush()

	fmt.Fscanf(in, "%d", &n)
	fmt.Fscanf(in, "%d", &m)
	fmt.Fscanln(in)

	// 创建树
	t := NewTree(n)
	nums := make([]int, n)

	for i := 0; i < n; i++ {
		var tmp int
		fmt.Fscanf(in, "%d", &tmp)
		nums[i] = tmp
	}
	t.build(1, 1, n, nums)
	fmt.Fscanln(in)

	// 读取命令
	for i := 0; i < m; i++ {
		var tmp int
		var x, y, z int
		fmt.Fscanf(in, "%d", &tmp)
		//fmt.Println("tmp", tmp)
		switch tmp {
		case 1:
			fmt.Fscanf(in, "%d", &x)
			fmt.Fscanf(in, "%d", &y)
			fmt.Fscanf(in, "%d", &z)
			fmt.Fscanln(in)
			t.update(1, x, y, z)
		case 2:
			fmt.Fscanf(in, "%d", &x)
			fmt.Fscanf(in, "%d", &y)
			fmt.Fscanln(in)
			fmt.Fprintf(out, "%d\n", t.query(1, x, y))
		}
	}
}
