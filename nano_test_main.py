import sys, os, traceback
sys.path.insert(0, os.getcwd())

try:
    import main_gui
    print('main_gui.py cargado OK')
    main_gui.main()
except Exception as e:
    print(f'ERROR: {e}')
    traceback.print_exc()
