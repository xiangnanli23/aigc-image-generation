import time
from functools import wraps

# get the date of the day like this format: 2020-12
def get_date(output_type: str = 'day') -> str:
    """
    output_type: year, month, day, hour, minute, second
    """
    if output_type == 'year':
        output_date = time.strftime("%Y", time.localtime())
    if output_type == 'month':
        output_date = time.strftime("%Y-%m", time.localtime())
    if output_type == 'day':
        output_date = time.strftime("%Y-%m-%d", time.localtime())
    if output_type == 'hour':
        output_date = time.strftime("%Y-%m-%d-%H", time.localtime())
    if output_type == 'minute':
        output_date = time.strftime("%Y-%m-%d-%H-%M", time.localtime())
    if output_type == 'second':
        output_date = time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime())
    return output_date

# get the date of the day like this format: 2020-12-31-12-59-59
def get_date_time() -> str:
    return time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime())


def timeit_decorator(func):
    """ calculate the time of executing the function """
    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()  # 记录开始时间
        result = func(*args, **kwargs)  # 执行被装饰的函数
        end_time = time.time()  # 记录结束时间
        execution_time = end_time - start_time  # 计算运行时间
        print(f"Function '{func.__name__}' executed in {execution_time:.4f} seconds")
        return result
    return wrapper




def main():
    # 使用装饰器
    @timeit_decorator
    def example_function(n):
        sum = 0
        for i in range(n):
            sum += i
        return sum

    # 调用被装饰的函数
    result = example_function(1000000)
    print(f"Result: {result}")




if __name__ == '__main__':
    # output_type = 'month'
    # print(get_date(output_type))
    # print(get_date_time())
    main()
    pass
    