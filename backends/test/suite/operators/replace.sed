s/self\, flow\: TestFlow/test_runner/g
s/self\._test_op/test_runner.lower_and_run_model/g
s/, flow//g
/@operator_test/d
/(OperatorTest):/d
s/dtype_test/parameterize_by_dtype/g
/flow,/d
/import TestFlow/d
/operator_test,/d
/OperatorTest,/d
