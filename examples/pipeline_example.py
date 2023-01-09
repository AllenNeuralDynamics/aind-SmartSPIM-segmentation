from aind_pipeline_repo_template.pipeline import Pipeline, PipelineStep

def main():
    # Creating pipeline steps

    def print_val(input, output, **param):
        print("exect: ", input, output, param)

    pip_step_1 = PipelineStep(
        "folder_input_1",
        "folder_output_1",
        executable=print_val,
        name="step_1",
    )

    pip_step_2 = PipelineStep(
        "folder_input_2",
        "folder_output_2",
        executable=print_val,
        name="step_2",
    )

    pipeline = Pipeline(
        "pipeline_1",
        "/home/camilo/repositories/aind-pipeline-repo-template/test_folder",
        [pip_step_1, pip_step_2],
    )

    args = {"step_1": {"h": 1, "c": 2}}

    pipeline.call(args)

    print(pipeline)


if __name__ == "__main__":
    main()
