def test_run(args):
    if args.microscope == "verios":
        params = {"ip": "192.168.0.1"}
        my_microscope = microscope.ThermoFisherVerios(params=params)
    elif args.microscope == "fake":
        params = {"W": 1024, "H": 1024, "dtype": np.uint16}
        my_microscope = microscope.FakeRandomMicroscope(params=params)
    elif args.microscope == "fake_data":
        params = {
            "images_ns": {
                50: "./examples/data/example1/loc_001_dwell_00050ns_00002_param_001_yi_1_xi_1_reg.png",
                100: "./examples/data/example1/loc_001_dwell_00100ns_00004_param_001_yi_1_xi_1_reg.png",
                200: "./examples/data/example1/loc_001_dwell_00200ns_00007_param_001_yi_1_xi_1_reg.png",
                500: "./examples/data/example1/loc_001_dwell_00500ns_00010_param_001_yi_1_xi_1_reg.png",
                1200: "./examples/data/example1/loc_001_dwell_01200ns_00014_param_001_yi_1_xi_1_reg.png",
            }
        }
        my_microscope = microscope.FakeDataMicroscope(params=params)
    else:
        raise ValueError("Unknown microscope type")

    if args.get_rescan_map == "test":
        params = {"type": "half", "fraction": 0.5}
        get_rescan_map = get_rescan_maps.GetRescanMapTest(params=params)
    elif args.get_rescan_map == "membrane_errors":
        params = {
            "do_clahe": False,
            "pad": 0,
            "search_step": 0.01,
            "rescan_prob": None,
            "rescan_p_thres": 0.5,
        }
        get_rescan_map = get_rescan_maps.GetRescanMapMembraneErrors(params=params)
    else:
        raise ValueError("Unknown get_rescan_map method")

    my_smart_em = smartem.SmartEM(
        microscope=my_microscope, get_rescan_map=get_rescan_map
    )
    print()
    print(my_smart_em)
    print()
    my_smart_em.initialize()
    params = {"fast_dwt": 50e-9, "slow_dwt": 500e-9, "plot": True}
    image = my_smart_em.acquire(params=params)
    my_smart_em.close()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", default="run", help="train the model")
    parser.add_argument(
        "--microscope",
        type=str,
        default="verios",
        help="Type of miroscope: verios, fake or fake_data",
    )
    parser.add_argument(
        "--get-rescan-map",
        type=str,
        default="membrane_errors",
        help="Method of rescan map generation: membrane_errors or test",
    )
    parser.add_argument("--verbose", type=int, default=1, help="verbosity level")
    args = parser.parse_args()
    assert args.mode in ["run", "test_run", "test", "train"]
    assert args.microscope in ["verios", "fake", "fake_data"]
    assert args.get_rescan_map in ["membrane_errors", "test"]

    from src import smartem
    from src.online import microscope
    from src.online import get_rescan_maps

    if args.mode == "run":
        raise NotImplementedError("TODO: add run script")
    elif args.mode == "test_run":
        test_run(args)
    elif args.mode == "test":
        raise NotImplementedError("TODO: add test script")
    elif args.mode == "train":
        raise NotImplementedError("TODO: add train script")
    else:
        raise ValueError("Unknown mode")
