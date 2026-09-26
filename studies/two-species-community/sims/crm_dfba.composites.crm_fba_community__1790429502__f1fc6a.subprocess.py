import json, os, sys, traceback
try:
    from crm_dfba.core import build_core
    from process_bigraph import Composite, gather_emitter_results
    try:
        from viva_emitters.sqlite_emitter import SQLiteEmitter
    except ImportError:  # process-bigraph < 1.4.17 (legacy location)
        from process_bigraph.emitter import SQLiteEmitter
    from process_bigraph.composite_generator import (
        _REGISTRY, build_generator, discover_generators,
        apply_core_extensions,
    )
    from vivarium_workbench.lib import composite_runs as cr
    from bigraph_schema.json_codec import BigraphJSONEncoder as _BJE
    _payload = {'spec_id': 'crm_dfba.composites.crm_fba_community', 'overrides': {}, 'run_id': 'crm_dfba.composites.crm_fba_community__1790429502__f1fc6a', 'db_file': '/Users/eranagmon/code/CRM_FBA--demo-investigation/studies/two-species-community/runs.db', 'steps': 360, 'emit_paths': [], 'default_emitter': 'sqlite', 'max_generations': 3, 'single_daughters': False, 'zarr_store': '/Users/eranagmon/code/CRM_FBA--demo-investigation/studies/two-species-community/runs.crm_dfba.composites.crm_fba_community__1790429502__f1fc6a.zarr'}
    if _payload['spec_id'] not in _REGISTRY: discover_generators()
    entry = _REGISTRY[_payload['spec_id']]
    core = build_core()
    core.register_link('SQLiteEmitter', SQLiteEmitter)
    # v2ecoli friction #16: register types/processes the composite
    # needs from packages build_core() doesn't know about (declared
    # via @composite_generator(core_extensions=[...])).
    core = apply_core_extensions(entry, core)
    doc = build_generator(entry, overrides=_payload['overrides'])
    state = doc.get('state', doc) if isinstance(doc, dict) else doc
    if _payload.get('emit_paths'):
        state = cr.inject_emitter_for_declared_paths(state, _payload['emit_paths'])
    # v2ecoli applicability gate (review CRITICAL 2): the xarray
    # study-run write path is wired ONLY for v2ecoli (its
    # colony/multigen loop in v2ecoli.library.xarray_run). Wiring
    # the generic flat-Step xarray emitter into composite_subprocess
    # is the DEFERRED Task 3 (it needs a division-aware drive loop).
    # So gate xarray on v2ecoli being importable, NOT on
    # default_emitter alone — otherwise a study run on a non-v2ecoli
    # workspace whose GLOBAL default is now "xarray" (Task 6) hits an
    # unconditional v2ecoli import and the subprocess ImportErrors.
    # A generic run instead falls through to the single-generation
    # sqlite branch below (writes sqlite, SUCCEEDS).
    try:
        import v2ecoli as _v2ecoli  # noqa: F401
        _v2ecoli_available = True
    except ImportError:
        _v2ecoli_available = False
    _mg = int(_payload.get('max_generations') or 1)
    _use_xarray = (
        _payload.get('default_emitter') == 'xarray' and _v2ecoli_available)
    # FU2b: generic SINGLE-generation xarray. A non-v2ecoli workspace
    # whose default resolves to xarray now writes a real zarr store
    # via the broker's flat-Step XArrayEmitter
    # (emitters.run_with_emitter), instead of silently falling back to
    # sqlite. SCOPE: single generation only (max_generations <= 1).
    # Multi-generation non-v2ecoli runs still take the sqlite path
    # below — there is NO generic division-aware xarray drive yet, so
    # retiring the gated v2ecoli xarray branch fully would need that
    # drive loop (a separate, larger feature).
    _use_generic_xarray = (
        _payload.get('default_emitter') == 'xarray'
        and not _v2ecoli_available and _mg <= 1)
    if (_payload.get('default_emitter') == 'xarray'
            and not _v2ecoli_available and not _use_generic_xarray):
        print('[xarray-run] multi-generation generic xarray awaits a '
              'division-aware drive; falling back to sqlite '
              '(v2ecoli not importable in this workspace).',
              file=sys.stderr)
    _view = []
    if _use_xarray:
        # Auto-view from the study's declared observables. v0 of
        # view_from_emit_paths is scalar-only — vector observables
        # (monomer_counts, fork_coordinates, RNAP_coordinates, …)
        # are skipped. If a study declares ONLY vector observables
        # (e.g. dnaa-01 emits only listeners.monomer_counts), the
        # auto-view is empty and the XArrayEmitter constructor
        # would crash. In that case, fall back to SQLite for this
        # run so the study isn't blocked. Defense-in-depth: even
        # with v2ecoli importable, guard the specific submodule
        # import so a packaging gap can't crash the run.
        try:
            from v2ecoli.library.xarray_run import (
                run_multigen_xarray, view_from_emit_paths,
            )
        except ImportError:
            print('[xarray-run] generic xarray study-runs await '
                  'Task 3; falling back to sqlite '
                  '(v2ecoli.library.xarray_run unavailable).',
                  file=sys.stderr)
            _use_xarray = False
    if _use_xarray:
        _view = view_from_emit_paths(_payload.get('emit_paths') or [])
        if not _view:
            print('[xarray-run] auto-view is empty (all declared '
                  'observables are vector / non-listeners-rooted); '
                  'falling back to SQLite emitter for this run.',
                  file=sys.stderr)
            _use_xarray = False
    if _use_xarray:
        # XArray multi-gen path: drive the composite externally past
        # divisions, per-generation emitter swap; results land in a
        # partitioned zarr store. See v2ecoli plan
        # 2026-05-12-migrate-emitters.md task 7.x.
        composite = Composite({'state': state}, core=core)
        _md = {
            'experiment_id': _payload['run_id'],
            'variant': 0,
            'lineage_seed': 0,
            'time_step': 1.0,
            'max_duration': float(_payload['steps']),
        }
        _xarr = run_multigen_xarray(
            composite,
            store_path=_payload['zarr_store'],
            view=_view,
            metadata_base=_md,
            max_steps=_payload['steps'],
            max_generations=_payload['max_generations'],
        )
        results = {'zarr_store': _xarr['store'],
                   'generations': _xarr['generations'],
                   'steps': _xarr['steps']}
    elif _use_generic_xarray:
        # FU2b: generic single-generation flat-Step XArray write.
        # Mirror run_runner.execute's run_with_emitter call, building
        # from the already-built state/core/emit_paths. The broker
        # injects XArrayEmitter as a flat Step, drives the composite,
        # writes a partitioned zarr store under out_dir, and
        # AUTO-FALLS-BACK to sqlite for an empty view (a composite
        # with no emittable observable) — so this NEVER blocks a run.
        from vivarium_workbench.lib import emitters as _emitters
        _out_dir = os.path.dirname(os.path.abspath(_payload['db_file']))
        # CRITICAL: pin the store to the STUDY convention
        # (<study>/runs.<run_id>.zarr == _payload['zarr_store']) so the
        # study read-path can find it. study_run_state.zarr_store_for_sim
        # resolves <db_stem>.<run_id>.zarr and study_charts globs
        # runs.*.zarr; the default <out_dir>/<run_id>.zarr name would
        # be unresolvable (None) → empty charts. store_path overrides it.
        _prov = _emitters.run_with_emitter(
            'xarray', state=state, run_id=_payload['run_id'],
            emit_paths=_payload.get('emit_paths') or [],
            out_dir=_out_dir, core=core, steps=_payload['steps'],
            db_file=_payload['db_file'], spec=None,
            store_path=_payload['zarr_store'])
        composite = _prov.get('composite')
        if _prov.get('output_kind') == 'zarr':
            results = {'zarr_store': _prov.get('store_path'),
                       'output_kind': 'zarr',
                       'steps': _prov.get('steps')}
        else:
            # Empty-view (or under-filled-buffer) auto-fall-back to
            # sqlite inside the broker — read the sqlite history.
            results = gather_emitter_results(composite)
    else:
        if _mg > 1 and _v2ecoli_available:
            # Multi-gen: workspace-side runner drives the
            # SQLiteEmitter externally (mirrors how the
            # xarray branch drives XArrayEmitter). The
            # composite does NOT get an injected emitter —
            # the static `agents/0/...` wiring would write
            # empty rows after division. The runner extracts
            # the followed agent's state each chunk and
            # calls `emitter.update` with it; on division it
            # switches to the daughter agent_id.
            composite = Composite({'state': state}, core=core)
            from v2ecoli.library.sqlite_run import run_multigen_sqlite
            _sq = run_multigen_sqlite(
                composite,
                run_id=_payload['run_id'],
                db_file=_payload['db_file'],
                emit_paths=_payload.get('emit_paths') or [],
                max_steps=_payload['steps'],
                max_generations=_mg,
                single_daughters=bool(_payload.get('single_daughters')),
                core=core,
            )
            results = {'steps': _sq['steps'],
                       'generations': _sq['generations']}
        else:
            state = cr.inject_sqlite_emitter(
                state, run_id=_payload['run_id'], db_file=_payload['db_file'])
            composite = Composite({'state': state}, core=core)
            cr.run_with_division(composite, _payload['steps'])
            results = gather_emitter_results(composite)

    # Flatten tuple keys to JSON-friendly dotted strings
    out = {}
    for path_tuple, entries in results.items():
        key = '.'.join(str(p) for p in path_tuple)
        out[key] = entries
    # Gather rendered viz HTML, if viva_superpowers is importable.
    viz_html = {}
    try:
        from process_bigraph.visualization import render_results
        rendered = render_results(composite)
        for path_tuple, payload in rendered.items():
            key = '.'.join(str(p) for p in path_tuple)
            viz_html[key] = payload
    except Exception:
        viz_html = {}
    # reproducible-rerun-spine Task 3 (G4), fix round 1: snapshot this
    # run's declared fingerprint_fields from the just-completed
    # composite's live state — only available HERE, in the child;
    # the parent (after this subprocess exits) reads it back via
    # result_fingerprint.fingerprint_run(). Best-effort: swallowed so
    # a snapshot failure never turns a successful run into a reported
    # @@@ERROR@@@.
    try:
        from vivarium_workbench.lib import result_fingerprint as _rfp
        _rfp.write_snapshot('/Users/eranagmon/code/CRM_FBA--demo-investigation/.pbg/runs/crm_dfba.composites.crm_fba_community__1790429502__f1fc6a', composite.state, [])
    except Exception:
        pass
    from bigraph_schema.json_codec import BigraphJSONEncoder as _BJE
    print('@@@RESULTS@@@')
    print(json.dumps({'results': out, 'viz_html': viz_html}, cls=_BJE))
except Exception as e:
    print('@@@ERROR@@@')
    print(traceback.format_exc())
