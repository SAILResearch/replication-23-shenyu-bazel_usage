# Replication package for the paper "Does Using Bazel Help Speed Up Continuous Integration Builds?"


### Abstract
A long continuous integration (CI) build forces developers to wait
for CI feedback before starting subsequent development activities, leading to
time wasted. In addition to a variety of build scheduling and test selection
heuristics studied in the past, new artifact-based build technologies like Bazel
have built-in support for advanced performance optimizations such as parallel
build and incremental build (caching of build results). However, little is known
about the extent to which new build technologies like Bazel deliver on their
promised benefits, especially for long-build duration projects.


In this study, we collected 383 Bazel projects from GitHub, 
then studied their usage of Bazel in popular CI services (GitHub Actions, CircleCI, Travis CI, or Buildkite), 
and compared the results with Maven projects. 
We conducted 14,150 experiments on 383 Bazel projects and analyzed the build logs of a subset of 70 buildable projects to evaluate the performance impact of Bazel's parallel builds. 
Additionally, we performed 145,900 experiments on the 70 buildable projects' last 100 commits to evaluate Bazel's incremental build performance. 
Our results show that 32.56\% of Bazel projects adopt a CI service but do not use Bazel in the CI service, while for those who do use Bazel in CI, 
26.36\% of them use other tools to facilitate Bazel's execution. Compared to sequential builds, 
the median speedups for long-build duration projects are 2.00x, 3.88, 7.36x, and 13.10x, at parallelism degrees 2, 4, 8, and 16, respectively, 
even though, compared to a clean build, applying incremental build achieves a median speedup of 4.22x (with the *Local-Deps-and-Results* strategy) 
and 4.73x (with the *Remote-Deps-and-Results* strategy) for long-build duration projects.
Our results provide guidance for developers to improve the usage of Bazel in their projects, 
and emphasize the importance of exploring modern build systems due to the current lack of literature and their potential advantages within contemporary software practices such as cloud computing and microservice.


### Project Structure

This replication package contains the following files:

- `analysis/`: The scripts used to analyze the projects' CI configuration files
- `data/`: The raw data of our study
- `experiments/`: The scripts used to run the experiments
- `projects/`: The scripts used to collect GitHub projects from Sourcegraph
- `visualization/`: The scripts used to preprocess and visualize the results

### Replication Steps

The codes have been tested on macOS 13.5 with Python 3.10 version.

#### Setup
```shell
# Create a virtual environment
python3 -m venv env
# Activate the virtual environment
source env/bin/activate
# Install the dependencies
pip install -r requirements.txt
```
#### Data Collection
We use the [Sourcegraph](https://sourcegraph.com/) API to identify the Bazel and Maven projects. Then,
employ the [GitHub API](https://docs.github.com/en/rest) to filter the projects that have less than 100 commits and stars.
Finally, we sample and clone the projects into the `repos/` directory.

The steps to collect the projects are as follows:

1. create a file `.github_token` in the root directory of this project, and put your GitHub token in it.
2. run the following codes in `main.py` to
   1. collect the Bazel and Maven projects from Sourcegraph,
   2. filter the projects that have less than 100 commits and stars,
   3. and clone the projects into the `repos/` directory.
```python3
from projects import project

if __name__ == '__main__':
    # ....
    projects = project.retrieve_projects()
```

The collected projects are stored in the `data/bazel_projects.csv`, `data/large_maven_projects_subset_projects.csv`,
and `data/small_maven_projects_subset_projects.csv` files.

#### RQ1
We analze the CI configuration files of the collected projects to answer RQ1. The steps to analyze the CI configuration files are as follows:

1. ensure you have the `repos/` directory that contains the collected projects.
2. run the following codes in `main.py` to
   1. identify the CI configuration files,
   2. extracted Bazel- and Maven-related shell commands from
      1. the CI configuration files,
      2. the shell scripts that are called by the CI configuration files,
      3. and the Make targets that are called by the CI configuration files.
   3. and extract cache related information from the CI configuration files,

```python3
from analysis import analysis

if __name__ == '__main__':
    # ....
    analysis.analyze("./repos")
```

The analysis results are stored in the `data/bazel-projects/`, `data/maven-large-projects/`,
   and `data/maven-small-projects/` directories.

#### RQ2
We perform experiments to answer RQ2. The experiments are run in Docker containers with the Docker version 20.10.24 and operating system Debian 11 (bullseye).

The steps to run the experiments are as follows:

**1. Build images**

Ensure you have the `experiments/parallelization/bazel_projects.csv` file that contains the collected projects.
Then, run the following commands to build Docker images for the experiments:

```shell
cd experiments/parallelization
# Build the experiment image (replace <parallel-experiment-runner-img> to your image name).
docker build -f Experiment.Dockerfile -t <parallel-experiment-runner-img> . 
# Build the experiment runner image (replace <parallel-experiement-runner-img> to your image name)
docker build -f Builder.Dockerfile -t <parallel-experiement> .
```

*optional*: push the images to a image registry if you want to run the experiments on a different machine.

**2. Run the experiments**

Run the following commands to start a experimental runner:
```shell
# start the experimental runner
docker run -d -v /var/run/docker.sock:/var/run/docker.sock -v /dev:/dev --name=parallel-experiment cizezsy/bazel_parallelization_experiment
```
Then, start the experiments by running the following commands within the experimental runner:
```shell
# enter the experimental runner
docker exec -it parallel-experiment bash
# run the experiments
bash start_experiments.sh
```
The experimental results are stored in the `/root/results.csv` file within the experimental runner.

#### RQ3

We used the same docker version and operating system as RQ2 to run the experiments for RQ3.


**1. Setup experiment environment**
Run the following commands to setup the machines for the experiments:
```shell
# create a directory to store the cloned git repositories
makdir -p /data/repo

# CI-Local cache
# mounting object storage buckets onto machine
# 1. create a directory to store the cached data
mkdir /cache_storage
# 2. install gcsfuse (if you use other object storage, please refer to the corresponding documentation)
export GCSFUSE_REPO=gcsfuse-`lsb_release -c -s`
echo "deb https://packages.cloud.google.com/apt $GCSFUSE_REPO main" | sudo tee /etc/apt/sources.list.d/gcsfuse.list
curl https://packages.cloud.google.com/apt/doc/apt-key.gpg | sudo apt-key add -
sudo apt-get update
sudo apt-get install gcsfuse
# 3. mount the object storage bucket to the directory
gcsfuse -o allow_other  -file-mode=777 -dir-mode=777 <your gcp bucket name> /cache_storage
```

Run the following commands to setup Bazel's remote cache (we use Google Cloud Storage as its storage backend):
```shell
docker pull buchgr/bazel-remote-cache
docker run -u 1000:1000 -v /path/to/cache/dir:/data \
	-p 9090:8080 -p 9092:9092 buchgr/bazel-remote-cache \
	--gcs_proxy.bucket <your gcp bucket name> \
	--gcs_proxy.use_default_credentials true \
	--gcs_proxy.json_credentials_file /path/to/your/credentials.json \
	--max_size 50
```

Then, update the codes in `experiments/cache/cache_experiments.py` to specify the cache server
```python
# update your cache servers abd ports here
cache_servers = ["cache-server_1", "cache-server_2", "cache-server_3", "cache-server_4", "cache-server_5"]
cache_server_ports = [-1, -1, -1, -1, -1]
```


**2. Build images**
```shell
cd experiments/cache
# Build the experiment image (replace <cache-experiment-runner-img> to your image name).
docker build -f Experiment.Dockerfile -t <cache-experiment-runner-img> .
# Build the experiment runner image (replace <cache-experiement-runner-img> to your image name)
docker build -f Builder.Dockerfile -t <cache-experiement> .
```
*optional*: push the images to a image registry if you want to run the experiments on a different machine.


**3. Run the experiments**

Run the following commands to start a experimental runner:
```shell
docker run -d -v /var/run/docker.sock:/var/run/docker.sock \
    -e GITHUB_API_TOKEN=<your github token> \
    -v /dev:/dev \
    --name=cache-experiment <cache-experiment-runner-img>
```

Then, start the experiments by running the following commands within the experimental runner:
```shell
# enter the experimental runner
docker exec -it cache-experiment bash
# run the experiments
bash start_experiments.sh
```

The experimental results are stored in the `/root/results.csv` file within the experimental runner. 
The build logs are stored in the `/root/build_logs/` directory within the experimental runner.

### Authors