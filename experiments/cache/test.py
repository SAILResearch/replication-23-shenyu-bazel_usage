import re


elapsed_time_matcher = re.compile(r"Elapsed time: (\d+\.\d+)s")
critical_path_matcher = re.compile(r"Critical Path: (\d+\.\d+)s")
remote_cache_hit_matcher = re.compile(r" (\d+) remote cache hit,")
disk_cache_hit_matcher = re.compile(r" (\d+) disk cache hit,")
total_processes_matcher = re.compile(r"INFO: (\d+) processes:")

if __name__ == "__main__":
    text = """
    Analyzing: target //:test_deps (1 packages loaded, 0 targets configured)
Analyzing: target //:test_deps (5 packages loaded, 5 targets configured)
Analyzing: target //:test_deps (5 packages loaded, 5 targets configured)
Analyzing: target //:test_deps (5 packages loaded, 5 targets configured)
Analyzing: target //:test_deps (5 packages loaded, 5 targets configured)
Analyzing: target //:test_deps (5 packages loaded, 5 targets configured)
Analyzing: target //:test_deps (46 packages loaded, 518 targets configured)
Analyzing: target //:test_deps (49 packages loaded, 665 targets configured)
INFO: Analyzed target //:test_deps (50 packages loaded, 834 targets configured).
INFO: Found 1 target...
[0 / 4] [Prepa] BazelWorkspaceStatusAction stable-status.txt
Target //:test_deps up-to-date:
  bazel-bin/libtest_deps.jar
INFO: Elapsed time: 120.496s, Critical Path: 2.93s
INFO: 4 processes: 3 disk cache hit, 1 internal.
INFO: Build completed successfully, 4 total actions
INFO: Build completed successfully, 4 total actions
    """

    build_log = text

    elapsed_time = None
    if match := elapsed_time_matcher.search(build_log):
        elapsed_time = float(match.group(1))

    critical_path = None
    if match := critical_path_matcher.search(build_log):
        critical_path = float(match.group(1))

    processes = None
    if match := total_processes_matcher.search(build_log):
        processes = int(match.group(1))


    if match := disk_cache_hit_matcher.search(build_log):
        cache_hit = int(match.group(1))
    else:
        cache_hit = 0


    if elapsed_time is None or critical_path is None or processes is None:
        result = "failed"