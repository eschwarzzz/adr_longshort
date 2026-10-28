while pgrep -f p4_borrow_fetch_ibd.py >/dev/null; do sleep 10; done
.venv/bin/python research/p4_borrow_fetch_ibd_universe.py
