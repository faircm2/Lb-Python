import subprocess, sys
CRLF = chr(13) + chr(10)
base = open('cap102_N100_gu1_45.py', 'rb').read().decode()          # flat start, 45 deg, ghost-velocity fix, tol 1e-4
anchor = "    u_ckl[:, :, 0] = -u_ckl[:, :, 1]; u_ckl[:, :, Yn+1] = -u_ckl[:, :, Yn]" + CRLF
assert base.count(anchor) == 1
log_lines = [
    "    import time as _t",
    "    global _T0",
    "    if iteration == 0: _T0 = _t.perf_counter()",
    "    _ux, _uy = u_ckl[0], u_ckl[1]",
    "    _dv = (np.roll(_ux, -1, 0) - np.roll(_ux, 1, 0)) / 2 + (np.roll(_uy, -1, 1) - np.roll(_uy, 1, 1)) / 2",
    "    with open('out/tolsens_TAG.csv', 'a') as _fh:",
    "        _fh.write('%d,%d,%.3e,%.2f,%.3e' % (iteration, ph_iter, np.max(epsilon), _t.perf_counter() - _T0, _dv[10:90, 2:30].std()) + chr(10))",
    "    if _t.perf_counter() - _T0 > 2700: raise SystemExit('wall limit')",
]
log = CRLF.join(log_lines) + CRLF
for tag in ['1e-3', '1e-4', '1e-5', '3e-6', '1e-6', '3e-7', '1e-7']:
    L = base.split(CRLF)
    i = [k for k, l in enumerate(L) if l.strip().startswith('epsilon_cutoff=1e-4') and 450 < k < 470][0]
    L[i] = L[i].replace('1e-4', tag)
    s = CRLF.join(L).replace(anchor, anchor + log.replace('TAG', tag))
    assert 'CORE_TOTAL_ITERATIONS = 20000' in s
    s = s.replace('CORE_TOTAL_ITERATIONS = 20000', 'CORE_TOTAL_ITERATIONS = 4500').replace('out/cap102_N100_gu1_45_it', 'out/tolsens_%s_it' % tag)
    fn = 'cap102_N100_tolsens_%s.py' % tag
    open(fn, 'wb').write(s.encode())
    subprocess.check_call([sys.executable, '-m', 'py_compile', fn])
    print('ok', fn)
