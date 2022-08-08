import os

os.system("nohup python evaluation_baseline_with_attack.py --is_event_trigger 1 --mtd_type 'robust' >robust_event.log & \
        nohup python evaluation_baseline_with_attack.py --is_event_trigger 0 --mtd_type 'robust' >robust_perio.log & \
        nohup python evaluation_baseline_with_attack.py --is_event_trigger 1 --mtd_type 'max_rank' >max_rank_event.log & \
        nohup python evaluation_baseline_with_attack.py --is_event_trigger 0 --mtd_type 'max_rank' >max_rank_perio.log & \
        nohup python evaluation_baseline_no_attack.py --mtd_type 'max_rank' >max_rank_no_attack.log & \
        nohup python evaluation_baseline_no_attack.py --mtd_type 'robust' >robust_no_attack.log") 