def solve_model(og,
                tol=1e-4,
                max_iter=1000,
                verbose=True,
                print_skip=25):
    """
    通过迭代贝尔曼算子求解

    """

    # 设置迭代循环
    v = og.u(og.grid)  # 初始条件
    i = 0
    error = tol + 1

    while i < max_iter and error > tol:
        v_greedy, v_new = T(v, og)
        error = np.max(np.abs(v - v_new))
        i += 1
        if verbose and i % print_skip == 0:
            print(f"第 {i} 次迭代的误差为 {error}。")
        v = v_new

    if error > tol:
        print("未能收敛！")
    elif verbose:
        print(f"\n在 {i} 次迭代后收敛。")

    return v_greedy, v_new
