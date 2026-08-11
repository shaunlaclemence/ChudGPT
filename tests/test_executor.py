from chudgpt._services.executor import ExecutorService


def test_run():
    exe = ExecutorService()
    res = exe.run("print(1 + 1)")
    assert res.stdout == "2\n"
    print(res)
