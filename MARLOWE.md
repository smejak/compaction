# Living doc on how to use Marlowe

Our group project id is **m000120-pm05**. So, in order to get any slurm command to run, you have to include the line
`-A marlowe-m000120-pm05 -p batch`

* Docs: https://docs.marlowe.stanford.edu/
* It seems like each user gets 32GB of memory.
    * You should instead go to the `/projects/m000120` folder for large storage needs.
      * You should be able to get there with `cd /projects/m000120`

## Quick command reference

| Command | What it does |
| --- | --- |
| `squeue -A marlowe-m000120-pm05` | Show all jobs on our project queue |
| `squeue -u $USER -o "%.18i %.9P %.50j %.8u %.2t %.10M %.6D %R"` | Show *your* current jobs |
| `scancel JOB_ID` | Cancel a job |
| `sinfo` | Show node statuses across partitions |
| `squeue -h -A marlowe-m000120-pm05 -o '%u %T' \| sort \| uniq -c` | Current running jobs in group, by user |
| `sreport cluster UserUtilizationByAccount -T gres/gpu Start=2026-02-01 End=2026-04-01 account=marlowe-m000120-pm05 -t hours` | Group GPU-hour usage |

# How to Run

Run workloads associated with your project in the batch partition and specify your project ID. For example, in your batch script:
```
#SBATCH -A marlowe-m000120-pm05
#SBATCH -p batch
```
Note that the batch partition has a maximum of 16-nodes per job. If you have jobs that require more than 16 nodes concurrently, we encourage you to request this explicitly as part of your proposal for an upcoming cycle so we can plan for and schedule it in advance.

# Marlowe Community Norms

1. Feel free to submit up to 8x jobs (inclusive) without authorization or notification.
2. For 9 or more jobs, please send a Slack message to `#sherlock-coordination` letting everyone know how many jobs you are running and time requested per job, and then submit your jobs. Afterwards, please actively monitor Slack for 30 minutes and be ready to cancel jobs as necessary to free up resources for others.
3. After 30 minutes, you don't have to be "on call" -- but please try to be a good GPU citizen and free up GPUs if people respond saying they need them.
4. Please only use Linderman Lab's Marlowe allocation for Linderman Lab work.

## Citing Marlowe

If you use Marlowe for a paper, you should cite it in that paper (typically in acknowledgements section).

Bibtex is

```
@misc{Kapfer2025Marlowe,
  author       = {Kapfer, Craig and Stine, Kurt and Narasimhan, Balasubramanian
                  and Mentzel, Christopher and Cand{\`e}s, Emmanuel},
  title        = {{Marlowe: Stanford's GPU-based Computational Instrument}},
  year         = {2025},
  howpublished = {Zenodo},
  doi          = {10.5281/zenodo.14751899},
  note         = {Version 0.1}
}
```

Citing may bring you to the attention of [Zoe Ryan](https://www.linkedin.com/in/zoe-ryan/), NVIDIA's higher ed rep for Stanford, which can lead you to getting free GPU hours from NVIDIA!!

# Quick start

Here is a quick way to verify that you have access to a GPU on marlowe:
```
srun -N 1 -G 1 -A marlowe-m000120-pm05 -p batch -t 00:30:00 --pty $SHELL
nvidia-smi
```

`sinfo` shows the statuses of the nodes.

`squeue` shows what jobs are running with what priority.

To view all jobs currently queued on our account (ie all members of Linderman Lab):

`squeue -A marlowe-m000120-pm05`.

And a command to see all of *your* current jobs:

`squeue -u $USER -o "%.18i %.9P %.50j %.8u %.2t %.10M %.6D %R"`.

# What partitions are available?

(info as of 3/6/26)

`sinfo` is helpful for seeing what state the different nodes are in (see the column `NODELIST`). Note that the 3 "partitions" are `hero`, `batch`, and `preempt`. However, if you look closely at the output of `sinfo`, you see that the nodes are the same in all of them! This overlap is because the partitions are really levels of priority.

The nodes are numbered `n01` to `n31`. Each node has 8 H100s. It can be useful for distributed training to request an entire node, with `-N 1 -G 8`. However, it will take slightly longer to launch your job in such a setting, because you have to wait for an entire node to be free.

```
(base) xavier18@login-01:~$ sinfo
PARTITION AVAIL  TIMELIMIT  NODES  STATE NODELIST
hero         up 1-00:00:00      1  inval n21
hero         up 1-00:00:00     16   mix- n[03-05,07-08,10,13-14,16-18,20,24,27-29]
hero         up 1-00:00:00      3 drain* n[01,15,26]
hero         up 1-00:00:00      7    mix n[02,06,09,11,23,30-31]
hero         up 1-00:00:00      4  alloc n[12,19,22,25]
batch        up 2-00:00:00      1  inval n21
batch        up 2-00:00:00     16   mix- n[03-05,07-08,10,13-14,16-18,20,24,27-29]
batch        up 2-00:00:00      3 drain* n[01,15,26]
batch        up 2-00:00:00      7    mix n[02,06,09,11,23,30-31]
batch        up 2-00:00:00      4  alloc n[12,19,22,25]
preempt      up   12:00:00      1  inval n21
preempt      up   12:00:00     16   mix- n[03-05,07-08,10,13-14,16-18,20,24,27-29]
preempt      up   12:00:00      3 drain* n[01,15,26]
preempt      up   12:00:00      7    mix n[02,06,09,11,23,30-31]
preempt      up   12:00:00      4  alloc n[12,19,22,25]
```
I think we are supposed to run on the `batch` partition.

On Stanford's Marlowe cluster, `preempt`, `batch`, and `hero` are SLURM partitions that dictate a job's priority, maximum runtime, and allocation requirements.

They all route to the exact same physical H100 nodes. The difference lies entirely in how the scheduler treats them:

### **`preempt` (Basic Access)**

* **Time Limit:** 12 hours.
* **Priority:** Lowest. Your job can be interrupted (preempted) within 15 minutes if a job in the `batch` or `hero` queues requests the node you are running on.
* **Access:** Available to anyone with basic Marlowe access. It does not require a special project allocation suffix.
* **Cost:** Lower subsidized rate ($0.20 per GPU-hour).

### **`batch` (Medium Projects)**

* **Time Limit:** 2 days (`2-00:00:00`).
* **Priority:** High. These are non-preemptible jobs; once your job starts, it is guaranteed those resources and will not be interrupted.
* **Access:** Requires an approved medium project allocation. You must append a medium project suffix to your account name (e.g., `marlowe-[ProjectID]-pm01`) in your SLURM script.
* **Cost:** Standard rate ($0.25 per GPU-hour).

### **`hero` (Large Projects)**

* **Time Limit:** 1 day (`1-00:00:00`).
* **Priority:** Highest. These are also non-preemptible but are prioritized to allow massive scaling across many nodes simultaneously.
* **Access:** Requires an approved large project allocation. You must append a large project suffix to your account name (e.g., `marlowe-[ProjectID]-pl01`) in your SLURM script.
* **Cost:** Standard rate ($0.25 per GPU-hour).

---

**In Short:** Submit to `preempt` for testing, debugging, or standard jobs if you don't mind the risk of interruption. Submit to `batch` or `hero` if you have the required project allocation and need guaranteed, uninterrupted compute time.

# Marlowe OnDemand

[Link](https://ood.marlowe.stanford.edu/pun/sys/dashboard)

* Can be used for VSCode forwarding or jupyter notebook
* Account: marlowe-m000120-pm05
* Partition: batch

# Getting Marlowe to email you

```
#SBATCH --mail-user=<sunetid>@stanford.edu
#SBATCH --mail-type=ALL
```

# Checking group usage

```
sreport cluster UserUtilizationByAccount -T gres/gpu Start=2026-02-01 End=2026-04-01 account=marlowe-m000120-pm05 -t hours
```

# Checking current running jobs in group
```
squeue -h -A marlowe-m000120-pm05 -o '%u %T' | sort | uniq -c
```

# Claude code on Marlowe

Run `curl -fsSL https://claude.ai/install.sh | bash`

Then just run `claude` and follow the prompts.

Finally, add this alias to your `.bash_aliases` file:

`alias claude="claude --dangerously-skip-permissions"`
