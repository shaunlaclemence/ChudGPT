from chudgpt.exceptions import ChudGPTInternalServerException, ServiceCode


def test_exception_print():
    try:
        raise ChudGPTInternalServerException("Internal Server Error", service_code=ServiceCode.FILE_SERVICE, error=NotImplementedError("Some value error"))
    except ChudGPTInternalServerException as err:
        print("\n")
        print(err)