def solve_model_time_iter(model,    # 含有模型信息的类
                          σ,        # 初始条件
                          tol=1e-4,
                          max_iter=1000,
                          verbose=True,
                          print_skip=25):

    # 设置迭代循环
    i = 0
    error = tol + 1

    while i < max_iter and error > tol:
        σ_new = K(σ, model)
        error = np.max(np.abs(σ - σ_new))
        i += 1
        if verbose and i % print_skip == 0:
            print(f"第 {i} 次迭代的误差为 {error}。")
        σ = σ_new

    if error > tol:
        print("未能收敛！")
    elif verbose:
        print(f"\n在 {i} 次迭代后收敛。")

    return σ_new
