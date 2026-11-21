"""
Builds a COPY of free_surface_SchanChen_nondim_D2Q9_08_Zhang_Cap02.py that additionally writes the raw fields
(phi, u_x, u_y, capillary force Fs, body force, rho) to <run folder>/fields_it<N>.npz at every iteration of interest
(the same iterations at which the script writes its phi_snapshot PNGs). The original script is not touched.
The copy writes into its own run folder (parameter stub + "__fields"), so the existing results are not overwritten.

usage:
    python make_field_dump_copy.py            -> Zhang_Cap02_fielddump.py   (same settings as Cap02: 300 x 300, 12001 its)
    python make_field_dump_copy.py --test     -> Zhang_Cap02_fielddump_test.py  (200 x 200, 400 its, for a quick check)
then run the written file like Cap02 itself (extra CLI options --tau_g --tau_f --vf_theta work as before).
"""
import re
import sys

SRC = "free_surface_SchanChen_nondim_D2Q9_08_Zhang_Cap02.py"
test = "--test" in sys.argv
dst = "Zhang_Cap02_fielddump_test.py" if test else "Zhang_Cap02_fielddump.py"

text = open(SRC, encoding="utf-8").read().replace("\r\n", "\n")


def replace_once(old, new):
    global text
    if text.count(old) != 1:
        raise SystemExit("cannot find exactly once (%d): %r" % (text.count(old), old))
    text = text.replace(old, new)


# own run folder
text, n = re.subn(r'(PARAMETER_STUB = .*str\(fc\.vf_theta\))\n', r'\1 + "__fields"\n', text)
if n != 1:
    raise SystemExit("PARAMETER_STUB line not found")
replace_once("default='phi_results.txt'", "default='phi_results_fields.txt'")

# the dump, right after the velocity update of the main loop
replace_once(
    "    u_ckl = zu_ckl(fc, z_fi, rho, body_force, _capillary_force)\n",
    "    u_ckl = zu_ckl(fc, z_fi, rho, body_force, _capillary_force)\n"
    "    if iteration in iterationsOfInterest:\n"
    "        np.savez_compressed(os.path.join(images_dir, 'fields_it%05d.npz' % iteration),\n"
    "                            phi=_phi.astype(np.float32), u=u_ckl.astype(np.float32),\n"
    "                            Fs=_capillary_force.astype(np.float32), Fb=body_force.astype(np.float32),\n"
    "                            rho=rho.astype(np.float32), iteration=iteration)\n")

if test:
    replace_once("DEFAULT_D_ND = 300\n", "DEFAULT_D_ND = 200\n")
    replace_once("CORE_TOTAL_ITERATIONS: int = 12001", "CORE_TOTAL_ITERATIONS: int = 400")

open(dst, "w", encoding="utf-8").write(text)
print("wrote", dst)
