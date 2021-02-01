"""
Demo
"""
import thorntw_mater.algo as tm

CONFIG_ALG = dict()

CONFIG_ALG = {
    "source_data_path": "input.xlsx",
    "LAT": 43,
    "SM": 250,
    "beta": 0.7,
    "SRT": -1,
    "dest_data_path": "out.xlsx",
    "dest_img_path": "img.png",
    "mean_calc": False
}

algo_status_out = tm.thorntw_mater_proc(
    source_path=CONFIG_ALG["source_data_path"],
    LAT=CONFIG_ALG["LAT"],
    SM=CONFIG_ALG["SM"],
    SRT=CONFIG_ALG["SRT"],
    beta=CONFIG_ALG["beta"],
    mean_calc=CONFIG_ALG["mean_calc"],
    file_out=CONFIG_ALG["dest_data_path"],
    img_out=CONFIG_ALG["dest_img_path"],
)

print(algo_status_out)
