# evaluation-tools
軌跡評価の為のライブラリ/コマンドラインツール

## Setup
```~\evaluation-tools> pip install -e .```

## Data Format
- trajectory data (est/gt)

    ```timestamp(index) x y yaw floor```

- eval-middle data

    - No tag or _rel

        ```timestamp(index) type value```
    
        ・ type : evaluation_type_tag ex) ce, ca, eag ...

        ・ value : evaluation value

    - _ble

        ```bdaddress(index) type value```
        
        ・ bdaddress : str BLEbeacon ID


## Commands
- 単体軌跡評価(timeline)

    ```
    do_eval -e [est_filename] -g [gt_filename]
            -s [evaluation_setting_filename] 

    options
    -e <name> estimated-trajectory filename (.csv)
    -g <name> ground-truth-trajectory filename (.csv)
    -s <name> evaluation_setting_filename (.json)
    -p <name> load est and gt trajectory from pickle (.pickle)

    Outputs/Returns
    output
        eval-middle-file "evaluation_result.csv"
    return
        eval-middle-df (pd.DataFrame)
    ```

- 相対軌跡評価(timeline)

    ```
    do_eval_rel -e1 [est_filename1] -g1 [gt_filename1]
                -e2 [est_filename2] -g2 [gt_filename2]
                -s [evaluation_setting_rel_filename]

    options
    -e1 <name> estimated-trajectory filename1 (.csv)
    -g1 <name> ground-truth-trajectory filename1 (.csv)
    -e2 <name> estimated-trajectory filename2 (.csv)
    -g2 <name> ground-truth-trajectory filename2 (.csv)
    -s <name> evaluation_setting_rel_filename (.json)
    -ed <dir> estimated-trajectory dirname (~\)
    -gd <dir> ground-truth-trajectory dirname (~\)

    Outputs/Returns
    output
        eval-middle-file "evaluation_rel_result.csv"
    return
        eval-middle-df (pd.DataFrame)
    ```

- eCDF描画

    ```
    plot_ecdf_from_csv -m [middle_filename] -t [tag_type]

    options
    -m <name> eval-middle-filename
    -t <name> evaluation_type_tag

    Outputs/Retruns
    output 
        eCDF graph image "eCDF_[tag_type].png"
    return
        None
    ```

- 軌跡総合評価

    ```
    show_result -m [middle_filename]

    options
    -m <name> eval-middle-filename

    Outputs/Returns
    output
        stdout eval_result_dictionary
            {[tag_type]:{avg:value,
                         median:value,
                         min:value,
                         max:value,
                         per50:value,
                         per75:value,
                         per90:value}, ...}
    
    return
        eval_result_dictonary

- BLE総合評価
    ```
    show_result_ble -m [middle_filename]

    options
    -m <name> eval-middle-filename_ble

    Outputs/Returns
    output
        stdout eval_result_dictionary
            {[tag_type]:{avg:value,
                         median:value,
                         min:value,
                         max:value,
                         per50:value,
                         per75:value,
                         per90:value}, ...}
    
    return
        eval_result_dictonary
    
- pickle -> csv est/gt file　抽出/変換
    ```
    extract_csv_from_pickle -p [pickle_filename] -t [target]

    options
    -p <name> pickle filename
    -t <name> dictonary key to extract to csv ex) pfloc, gt

    Outputs/Returns
    outputs
        csv file. Format : [timestamp(index), x, y, yaw, floor]

    returns
        None

