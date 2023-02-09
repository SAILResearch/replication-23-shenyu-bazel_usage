import subprocess

# string list
critical_path_list_ctj = ["//agent/src/main/java/com/code_intelligence/jazzer/runtime:coverage_map_remove_this_part_",
                          "//agent/src/main/java/com/code_intelligence/jazzer/runtime:runtime",
                          "//agent/src/main/java/com/code_intelligence/jazzer/instrumentor:instrumentor",
                          "////agent/src/main/java/com/code_intelligence/jazzer/agent:agent_installer",
                          "//driver/src/main/java/com/code_intelligence/jazzer/driver:fuzz_target_runner_remove_this_part_",
                          "//driver/src/main/java/com/code_intelligence/jazzer/driver:driver",
                          "//driver/src/main/java/com/code_intelligence/jazzer:jazzer_lib",
                          "//driver/src/main/java/com/code_intelligence/jazzer:jazzer_unshaded",
                          "//driver/src/main/java/com/code_intelligence/jazzer:jazzer_standalone",
                          "//:jazzer_release"]

critical_path_openexr = ["//:OpenEXR"]


def analyze_CodeIntelligenceTesting_jazzer_targets(directory):
    output = subprocess.check_output("bazel query //...", shell=True, text=True, cwd=directory)
    targets = output.split("\n")

    with open("../data/targets_ctj_analysis.csv", "w") as f:
        f.write(
            "target,in_degree,out_degree,number_of_dependencies,number_of_dependents,num_of_source_files,critical_path\n")
        for t in targets:
            in_degree = subprocess.check_output(
                f"bazel query 'deps({t},1)' --output maxrank | awk '($1 < 5) {{ print $2;}} ' | wc -l",
                shell=True, cwd=directory, text=True).strip()


            out_degree = subprocess.check_output(
                f"bazel query --universe_scope=//... --order_output=no 'allrdeps({t},1)'| wc -l",
                shell=True, cwd=directory, text=True).strip()
            num_of_transitive_deps = subprocess.check_output(
                f"bazel query 'deps({t})' --output maxrank | awk '($1 < 5) {{ print $2;}} ' | wc -l",
                shell=True, cwd=directory, text=True).strip()
            num_of_transitive_rdeps = subprocess.check_output(
                f"bazel query --universe_scope=//... --order_output=no 'allrdeps({t})'| wc -l",
                shell=True, cwd=directory, text=True).strip()
            num_of_source_files = subprocess.check_output(
                f"bazel query 'kind(\"source file\", deps({t}))'| grep \"^//\" | wc -l",
                shell=True, cwd=directory, text=True).strip()
            critical = t in critical_path_list_ctj
            f.write(
                f"{t},{in_degree},{out_degree},{num_of_transitive_deps},{num_of_transitive_rdeps},{num_of_source_files},{critical}\n")


def analyze_openexr_targets(directory):
    output = subprocess.check_output("bazel query //...", shell=True, text=True, cwd=directory)
    targets = output.split("\n")

    with open("../data/targets_openexr_analysis.csv", "w") as f:
        f.write(
            "target,in_degree,out_degree,number_of_dependencies,number_of_dependents,num_of_source_files,critical_path\n")
        for t in targets:
            in_degree = subprocess.check_output(
                f"bazel query 'deps({t},1)' --output maxrank | awk '($1 < 5) {{ print $2;}} ' | wc -l",
                shell=True, cwd=directory, text=True).strip()


            out_degree = subprocess.check_output(
                f"bazel query --universe_scope=//... --order_output=no 'allrdeps({t},1)'| wc -l",
                shell=True, cwd=directory, text=True).strip()
            num_of_transitive_deps = subprocess.check_output(
                f"bazel query 'deps({t})' --output maxrank | awk '($1 < 5) {{ print $2;}} ' | wc -l",
                shell=True, cwd=directory, text=True).strip()
            num_of_transitive_rdeps = subprocess.check_output(
                f"bazel query --universe_scope=//... --order_output=no 'allrdeps({t})'| wc -l",
                shell=True, cwd=directory, text=True).strip()
            num_of_source_files = subprocess.check_output(
                f"bazel query 'kind(\"source file\", deps({t}))'| grep \"^//\" | wc -l",
                shell=True, cwd=directory, text=True).strip()
            critical = t in critical_path_openexr
            f.write(
                f"{t},{in_degree},{out_degree},{num_of_transitive_deps},{num_of_transitive_rdeps},{num_of_source_files},{critical}\n")


analyze_CodeIntelligenceTesting_jazzer_targets(
    "/Users/zhengshenyu/GolandProjects/bazel-testing-practices/repos/CodeIntelligenceTesting_jazzer")
# analyze_openexr_targets(
#     "/Users/zhengshenyu/GolandProjects/bazel-testing-practices/repos/AcademySoftwareFoundation_openexr")
