class RequestError(Exception): ...
class InferenceError(Exception): ...
class ThreadOccupancyError(Exception): ...



def raise_error():
    raise InferenceError(f"{InferenceError.__name__}: there's wrong with image!")

def main():
    try:
        raise_error()
    except Exception as e:
        print(e)
        print(type(e))


if __name__ == '__main__':
    main()
