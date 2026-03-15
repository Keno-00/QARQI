import cv2, math
import matplotlib.pyplot as plt
import utils
import circuit
import numpy as np
import plots
from pathlib import Path
from datetime import datetime
import csv

CSV_PATH = Path("qarqi_runs.csv")

def save_rows_to_csv(rows, csv_path=CSV_PATH):
    fieldnames = [
        "timestamp", "n", "bins", "shots", "shots_per_bin",
        "mse", "psnr"
    ]
    write_header = not csv_path.exists()
    with csv_path.open("a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        if write_header:
            w.writeheader()
        w.writerows(rows)
def main(shots = 1000,n = 4):

    myimg = cv2.imread("lenna.jpg")
    myimg = cv2.resize(myimg, (n,n))
    myimg = cv2.cvtColor(myimg,cv2.COLOR_RGB2GRAY)
    d = round((myimg.shape[1])/2)
    N = myimg.shape[1]
    angle_norm = utils.angle_map(myimg)
    H,W = angle_norm.shape
    pol_mag_matrix = []

    for r,c in np.ndindex(H,W):
            cord = utils.compute_register(N,r,c)
            pol_mag_matrix.append(cord)

    qc,reg = circuit.QARQI_init(d)
    data_qc = circuit.QARQI_upload_intensity(qc,reg,N,pol_mag_matrix,angle_norm)
    counts,sv = circuit.QARQI_simulate(data_qc,shots)

    bins = utils.make_bins(counts=counts,d=d)
    #plots.plot_hits_grid(bins,d,kind="hit")
    #plots.plot_hits_grid(bins,d,kind="miss")
    grid = plots.bins_to_grid(bins,d,kind="p")
    #grid,_,_,_=plots.plot_hits_grid(bins,d,kind="p")
     
    #plots.plot_hits_scatter(bins,d,kind="hit")
    #plots.plot_hits_scatter(bins,d,kind="miss")
    #plots.plot_hits_scatter(bins,d,kind="p")
    
    newimg = plots.grid_to_image_uint8(grid,d,0.0,1.0)
    #plots.show_image_comparison(myimg,newimg)
    return myimg,newimg

#

if __name__ == "__main__":
    n = 8
    bin_of_n = 2*(n**2)
    tests = 100
    do_tests = False
    shots = [100]
    run_psnr = []
    run_mse = []
    rows = []

    if do_tests:
        for j in range(2, tests):
            shots.append(bin_of_n * j)

    for i in shots:
        print(f"Image size: {n}x{n}\nBins: {bin_of_n}\nCurrent Shots: {i}\nShots per Bin: {i/bin_of_n}")
        gt_img, rec_img = main(i, n)

        #plots.plot_mse_map(gt_img, rec_img)
        #plots.plot_psnr_map(gt_img, rec_img)

        i_mse = plots.compute_mse(gt_img, rec_img)
        i_psnr = plots.compute_psnr(gt_img, rec_img)

        run_mse.append(i_mse)
        run_psnr.append(i_psnr)

        rows.append({
            "timestamp": datetime.now().isoformat(timespec="seconds"),
            "n": n,
            "bins": bin_of_n,
            "shots": i,
            "shots_per_bin": i / bin_of_n,
            "mse": float(i_mse),
            "psnr": float(i_psnr),
        })

        print(f'===================\nCurrently running: {i} shots \nCurrent MSE: {i_mse}\nCurrent PSNR: {i_psnr}\n===================\n')
    save_rows_to_csv(rows)

    plots.plot_shots_vs_mse(shots, run_mse)
    plots.plot_shots_vs_psnr(shots, run_psnr)
