## Resources

### model.pte
- Internally generated after D62209852, 2024-09-06 with:
    ```
    buck2 run fbcode//executorch/examples/portable/scripts:export -- --model_name="add"
    ```
- In OSS, the same file can be generated after [#5145](https://github.com/pytorch/executorch/pull/5145), 2024-09-06 with:
    ```
    python -m examples.portable.scripts.export --model_name="add"
    ```
