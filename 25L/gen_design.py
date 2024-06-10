import math, argparse, random
import numpy as np
import gzip, pickle
import json
from scipy.integrate import quad
from scipy import optimize


def get_bij_from_epsij(e, r0=1.0, rc=3.0, beta=1.0):
    def lj126_B2_integrand(x, e, beta=1.0):
        return x**2 * (1 - np.exp(-beta * e * 4 * (pow(1 / x, 12) - pow(1 / x, 6))))

    return quad(lj126_B2_integrand, r0, rc, args=(e, beta))[0]


def get_epsij_from_bij(bij):
    """eps_ij is in the unit of beta"""

    def objective(eps_ij):
        return (bij - get_bij_from_epsij(eps_ij)) ** 2

    res = optimize.minimize(objective, 1.0)
    return res.x


def get_LJparameters(u, scale=1.0):
    """
    return: nonlinear mapping of LJ  parameters
    """
    u = u / np.mean(u)  # normalize u
    b = np.array(u) * scale  # get bij scaled by constant (input)
    eps_LJ = np.zeros(u.shape)
    for i in range(u.shape[0]):
        for j in range(i, u.shape[1]):
            eps_LJ[i, j] = eps_LJ[j, i] = get_epsij_from_bij(b[i, j])
    return -eps_LJ


def makeseq(row):
    """
    make the sequence less blocky by interleaving different monomers
    """
    cnts = {}
    for i, cnt in enumerate(row):
        cnts[i] = cnt
    pos = {}
    for m in cnts:
        if cnts[m] == 1.0:
            pos[m] = [1 / 2.0]
        else:
            pos[m] = np.arange(1, cnts[m] + 1) / (1 + cnts[m])
    locs, types = [], []
    for m in cnts:
        for j in range(len(pos[m])):
            locs.append(pos[m][j])
            types.append(m)
    mask = np.argsort(np.array(locs))
    return "".join([str(types[_] + 1) for _ in mask])


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(
        dest="mode", required=True, help="target-generation mode"
    )
    parser_frk = subparsers.add_parser(
        "fullrank", help="full rank eps with default design solution"
    )
    parser_frk.add_argument(
        "path", type=str, help=".p.gz file containing eps solution from convx solver"
    )

    parser_lrk = subparsers.add_parser(
        "lowrank", help="low rank solution from nmf design"
    )
    parser_lrk.add_argument(
        "path", type=str, help=".p.gz file containing solution to the nmf problem"
    )
    parser.add_argument("Lx", type=float, help="target Lx/Ly for the compressed box")
    parser.add_argument("Lz", type=float, help="target Lz for the compressed box")
    parser.add_argument(
        "output_summary", type=str, help=".json summary file for all phases"
    )
    parser.add_argument("output_paircoeff", type=str, help=".dat file containing u_ij")
    parser.add_argument(
        "--phi", type=float, default=0.5, help="target compressed volume fraction [0.5]"
    )
    parser.add_argument("--nx", type=int, default=6, help="number of x grid [6]")
    parser.add_argument("--ny", type=int, default=6, help="number of y grid [6]")
    parser.add_argument("--nz", type=int, default=12, help="number of zgrid [12]")
    parser.add_argument(
        "--eps-scale", type=float, default=0.4, help="scale on avergae eps [0.4]"
    )

    clargs = parser.parse_args()

    summary = {}
    volume = clargs.Lx**2 * clargs.Lz
    vi = np.pi / 6  # sigma = 1 for LJ beads
    landscape_test = None
    if clargs.mode == "fullrank":
        with gzip.open(clargs.path, "rb") as f:
            results = pickle.load(f)
            L = int(results["L"])
            K, N = results["targets"].shape
            targets = results["targets"]
            W, u = np.diag([L] * N), -results["eps"].matrix() / L**2
            landscape_test = (
                results["test_results_targets"]["success"]
                and results["test_results_dilute"]["success"]
            )

    if clargs.mode == "lowrank":
        with gzip.open(clargs.path, "rb") as f:
            results = pickle.load(f)
        targets = results["problem"]["targets"]
        W, u = results["res"]["W"], results["res"]["Y"]
        try:
            landscape_test = (
                results["test_results_targets"]["success"]
                and results["test_results_dilute"]["success"]
            )
        except:
            print("Skippng...")
    if not landscape_test:
        print("WARNING: landscape test failed")

    print("W=\n", W)
    print("u=\n", u)
    N, r = W.shape
    summary["N"] = N
    summary["r"] = r
    phisp = results["test_results_targets"]["phisp"]
    K = phisp.shape[0]
    all_sequences = []
    components = {}
    for k in range(N):
        row = W[k]
        seq = makeseq(row)
        print(seq)
        all_sequences.append(seq)
        components[seq] = 1 + k

    # generate phases
    phases = {0: {}}

    # dilute phase
    for c in range(N):
        phases[0][all_sequences[c]] = 6.0

    # condensed phases
    for alpha in range(K):
        phases[1 + alpha] = {}
        checkn = 0
        for c in range(N):
            if targets[alpha][c] > 0:
                x = targets[alpha][c] / targets[alpha].sum()
                n_ = np.floor(clargs.nx * clargs.ny * clargs.nz * x)
                checkn += n_ * len(all_sequences[c])
                phases[1 + alpha][all_sequences[c]] = n_
                print(x, n_)
        print(phases[1 + alpha])
        print("volume fraction: ", checkn * vi / volume)

    summary["components"] = components
    summary["phases"] = phases
    summary["target_box_dim"] = (clargs.Lx, clargs.Lx, clargs.Lz)
    with open(clargs.output_summary, "w") as pout:
        json.dump(summary, pout, indent=2)

    # generate pair coefficents
    u_eff = get_LJparameters(u, clargs.eps_scale)
    with open(clargs.output_paircoeff, "w") as fout:
        for i in range(r):
            for j in range(i, r):
                fout.write(
                    "pair_coeff      %d %d %.3f 1.0 3.0 \n"
                    % (i + 1, j + 1, u_eff[i, j])
                )
