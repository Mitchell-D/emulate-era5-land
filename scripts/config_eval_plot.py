config = {
    "EvalJointHist":{
        ## --------------------------------------------------------- ##
        ## temperature / snow histogram
        ## --------------------------------------------------------- ##

        "tmp-snow":{"err-diff":{
            ## swm-7 absolute error
            "swm-7-abs":{"plot_type":"hist-cov",
                "cov_feats":[("err-abs","diff swm-7")],
                "cov_metric":"cov_mean",
                "plot_spec":{
                    "hist_title":"swm-7 MAE wrt temperature and swe",
                    "xlabel":"Snow Water Equivalent (m^3/m^3)",
                    "ylabel":"Temperature (K)",
                    "cov_title":"Mean Absolute Error in 0-7cm Layer"
                    }},

            ## swm-7 error bias
            "swm-7-bias":{"plot_type":"hist-cov",
                "cov_feats":[("err-bias","diff swm-7")],
                "cov_metric":"cov_mean",
                "plot_spec":{}},
            }},

        ## --------------------------------------------------------- ##
        ## plant trans / evapotranspiration histogram
        ## --------------------------------------------------------- ##
        "trsp-evp":{"err-diff":{
            ## swm-7 absolute error
            "swm-7-abs":{"plot_type":"hist-cov",
                "cov_feats":[("err-abs","diff swm-7")],
                "cov_metric":"cov_mean",
                "plot_spec":{
                    "hist_title":"swm-7 MAE wrt plant transpiration and " + \
                            "total evapotranspiration",
                    "xlabel":"Plant Transpiration",
                    "ylabel":"Evapotranspiration",
                    "cov_title":"Mean Absolute Error in 0-7cm Layer"
                    }},

            ## swm-7 error bias
            "swm-7-bias":{"plot_type":"hist-cov",
                "cov_feats":[("err-bias","diff swm-7")],
                "cov_metric":"cov_mean",
                "plot_spec":{
                    "hist_title":"swm-7 bias wrt plant transpiration and " + \
                            "total evapotranspiration",
                    "xlabel":"Plant Transpiration",
                    "ylabel":"Evapotranspiration",
                    "cov_title":"Mean Absolute Error in 0-7cm Layer"
                    }},

            ## swm-28 absolute error
            "swm-28-abs":{"plot_type":"hist-cov",
                "cov_feats":[("err-abs","diff swm-28")],
                "cov_metric":"cov_mean",
                "plot_spec":{
                    "hist_title":"swm-28 MAE wrt plant transpiration and " + \
                            "total evapotranspiration",
                    "xlabel":"Plant Transpiration",
                    "ylabel":"Evapotranspiration",
                    "cov_title":"Mean Absolute Error in 0-7cm Layer"
                    }},

            ## swm-28 error bias
            "swm-28-bias":{"plot_type":"hist-cov",
                "cov_feats":[("err-bias","diff swm-28")],
                "cov_metric":"cov_mean",
                "plot_spec":{
                    "hist_title":"swm-28 bias wrt plant transpiration and " + \
                            "total evapotranspiration",
                    "xlabel":"Plant Transpiration",
                    "ylabel":"Evapotranspiration",
                    "cov_title":"Mean Absolute Error in 0-7cm Layer"
                    }},
            }},

        ## --------------------------------------------------------- ##
        ## temperature/dewpoint histogram wrt diff error
        ## --------------------------------------------------------- ##
        "tmp-dwpt":{"err-diff":{
            ## swm-7 absolute error
            "swm-7-abs":{"plot_type":"hist-cov",
                "cov_feats":[("err-abs","diff swm-7")],
                "cov_metric":"cov_mean",
                "plot_spec":{
                    "hist_title":"0-7cm counts wrt temp/dewpoint",
                    "xlabel":"dewpoint (K)", "ylabel":"temperature (K)",
                    "cov_title":"0-7cm increment mae",
                    "cov_vmin":0,
                    "cov_vmax":0.001,
                    "diagonal_width":2,
                    "aspect":1,
                    }, "plot_params":{"plot_diagonal":True}},
            ## swm-7 absolute error variance
            "swm-7-abs-stddev":{"plot_type":"hist-cov",
                "cov_feats":[("err-abs","diff swm-7")],
                "cov_metric":"cov_stddev",
                "plot_spec":{
                    "hist_title":"0-7cm counts wrt temp/dewpoint",
                    "xlabel":"dewpoint (K)", "ylabel":"temperature (K)",
                    "cov_title":"0-7cm increment stddev of mae",
                    "cov_vmin":0,
                    "cov_vmax":0.005,
                    "aspect":1,
                    "diagonal_width":2,
                    }, "plot_params":{"plot_diagonal":True}},
            ## swm-7 error bias
            "swm-7-bias":{"plot_type":"hist-cov",
                "cov_feats":[("err-bias","diff swm-7")],
                "cov_metric":"cov_mean",
                "plot_spec":{
                    "hist_title":"0-7cm counts wrt temp/dewpoint",
                    "xlabel":"dewpoint (K)", "ylabel":"temperature (K)",
                    "cov_title":"0-7cm increment bias wrt temp/dewpoint",
                    "cov_vmin":-.0005,
                    "cov_vmax":.0005,
                    "cov_cmap":"RdBu",
                    "aspect":1,
                    "diagonal_width":2,
                    }, "plot_params":{"plot_diagonal":True}},
            ## swm-28 absolute error
            "swm-28-abs":{"plot_type":"hist-cov",
                "cov_feats":[("err-abs","diff swm-28")],
                "cov_metric":"cov_mean",
                "plot_spec":{
                    }},
            ## swm-28 error bias
            "swm-28-bias":{"plot_type":"hist-cov",
                "cov_feats":[("err-bias","diff swm-28")],
                "cov_metric":"cov_mean",
                "plot_spec":{
                    }},
            }},

        ## --------------------------------------------------------- ##
        ## inc. change vs true state histogram w error covarites
        ## --------------------------------------------------------- ##
        "state-diff-swm-7":{"err-diff":{
            ## error bias
            "bias":{"plot_type":"hist-cov",
                "cov_feats":[("err-bias","diff swm-7")],
                "cov_metric":"cov_mean",
                "plot_spec":{}},

            ## absolute error
            "abs":{"plot_type":"hist-cov",
                "cov_feats":[("err-abs","diff swm-7")],
                "cov_metric":"cov_mean",
                "plot_spec":{}},
            }},
        "state-diff-swm-28":{"err-diff":{
            ## error bias
            "bias":{"plot_type":"hist-cov",
                "cov_feats":[("err-bias","diff swm-28")],
                "cov_metric":"cov_mean",
                "plot_spec":{}},

            ## absolute error
            "abs":{"plot_type":"hist-cov",
                "cov_feats":[("err-abs","diff swm-28")],
                "cov_metric":"cov_mean",
                "plot_spec":{}},
            }},
        "state-diff-swm-100":{"err-diff":{
            ## error bias
            "bias":{"plot_type":"hist-cov",
                "cov_feats":[("err-bias","diff swm-100")],
                "cov_metric":"cov_mean",
                "plot_spec":{}},

            ## absolute error
            "abs":{"plot_type":"hist-cov",
                "cov_feats":[("err-abs","diff swm-100")],
                "cov_metric":"cov_mean",
                "plot_spec":{}},
            }},
        "state-diff-swm-289":{"err-diff":{
            ## error bias
            "bias":{"plot_type":"hist-cov",
                "cov_feats":[("err-bias","diff swm-289")],
                "cov_metric":"cov_mean",
                "plot_spec":{}},

            ## absolute error
            "abs":{"plot_type":"hist-cov",
                "cov_feats":[("err-abs","diff swm-289")],
                "cov_metric":"cov_mean",
                "plot_spec":{}},
            }},

        ## --------------------------------------------------------- ##
        ## validation histograms
        ## --------------------------------------------------------- ##
        "swm-7":{"counts":{"vc":{"plot_type":"hist", "plot_spec":{}}}},
        "swm-28":{"counts":{"vc":{"plot_type":"hist", "plot_spec":{}}}},
        "swm-100":{"counts":{"vc":{"plot_type":"hist", "plot_spec":{}}}},
        "swm-289":{"counts":{"vc":{"plot_type":"hist", "plot_spec":{}}}},

        ## --------------------------------------------------------- ##
        ## differential validation histograms
        ## --------------------------------------------------------- ##
        "diff-swm-7":{"counts":{"vc":{"plot_type":"hist", "plot_spec":{}},}},
        "diff-swm-28":{"counts":{"vc":{"plot_type":"hist", "plot_spec":{}},}},
        "diff-swm-100":{"counts":{"vc":{"plot_type":"hist", "plot_spec":{}},}},
        "diff-swm-289":{"counts":{"vc":{"plot_type":"hist", "plot_spec":{}},}},
        }
    }

