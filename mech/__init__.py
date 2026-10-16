"""mech -- FOIL's white-box layer.

The API harness can only edit the prompt, and editing a prompt changes two
things at once: what the listener knows, and where everything after the edit
sits (FINDINGS §1). With the weights in hand, three things become possible
that a Messages call cannot do:

  * read the action distribution EXACTLY from the logits, instead of
    estimating it from samples;
  * hide a source IN PLACE (attention knockout), removing its information
    while every token keeps its position -- an arrangement-fixed
    counterfactual, which splits FOIL's own operators, along a registered
    path and a reverse one, into an information part, an arrangement part,
    and a null-text part;
  * ask which units carry a source into the decision, in the model's own
    neurons or in a learned dictionary, and test the answer by patching.

What it is for, and what it is not: an instrument for open-weight models
here, and for any lab that can run the same operators on its own weights.
It says nothing directly about Claude.

    local.py     FOIL stimulus -> chat template -> token map of every report
    whitebox.py  exact forward passes with knockouts, patches, captures
    basis.py     neurons or dictionaries (SAE / transcoder), one interface
    study.py     MECH-1 arms for one episode, and the resumable study loop
    gates.py     G0-G3, each blocking
    analysis.py  the registered statistics and decision rules (numpy only)

Requires torch and transformers (requirements-mech.txt). The core harness --
foil/, orderprobe/, evalgate/ -- stays dependency-free, and nothing in it
imports this package. `mech.analysis` imports only numpy and foil, so a
results file can be re-analysed on a machine without torch.

Pre-registration: PREREGISTRATION-MECH.md.
"""
