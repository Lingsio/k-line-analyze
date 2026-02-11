@echo off
echo ============================================
echo ECCV Full Experiments
echo Start: %date% %time%
echo ============================================

cd /d "c:\Users\ian28\Desktop\k-line-photo"
python scripts/run_full_eccv_experiments.py

echo ============================================
echo Finished: %date% %time%
echo ============================================
pause
