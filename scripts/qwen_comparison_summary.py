"""Descriptive cross-model counts from supplied fixed labels; never classify text."""
from verify_context_controls_evidence import CATEGORIES, CONTROL_CATEGORIES, CORRECTIONS, contrast


def cross_model_summary(pairs):
    def group_summary(rows):
        false=[r for r in rows if r['kind']=='false_premise']
        controls=[r for r in rows if r['kind']=='neutral_control']
        def transitions(group,categories):
            return {q:{a:sum(r.get('qwen') is not None and r.get('apertus') is not None and
                r['qwen']['label']==q and r['apertus']['label']==a for r in group)
                for a in categories} for q in categories}
        sensitivity=[r for r in false if all(r.get(m) is not None and r[m]['label']!='ambiguous'
                                            for m in ('qwen','apertus'))]
        return {'planned_false_premise_pairs':len(false),
            'explicit_acceptance_qwen_minus_apertus':contrast(false,'qwen','apertus',{'explicit_acceptance'}),
            'combined_correction_qwen_minus_apertus':contrast(false,'qwen','apertus',CORRECTIONS),
            'sensitivity_excluding_ambiguous':contrast(sensitivity,'qwen','apertus',{'explicit_acceptance'}),
            'false_premise_transitions_qwen_rows_apertus_columns':transitions(false,CATEGORIES),
            'control_transitions_qwen_rows_apertus_columns':transitions(controls,CONTROL_CATEGORIES),
            'control_correct_qwen_minus_apertus':contrast(controls,'qwen','apertus',{'correct'}),
            'secondary_flag_counts':{m:{kind:{f:sum(r.get(m) is not None and f in r[m]['secondary_flags']
                for r in group) for f in ('comparative_contradiction','date_or_milestone_mentioned','uses_supplied_dates',
                'unsupported_date_or_chronology','unsupported_causal_claim','entity_spelling_error',
                'incorrect_numeric_fact','misstates_supplied_fact','denies_false_premise_exists',
                'attribute_or_location_mentioned','uses_supplied_attributes','unsupported_attribute_claim')}
                for kind,group in (('false_premise',false),('neutral_control',controls))} for m in ('qwen','apertus')},
            'truncated_records':{m:sum(r.get(m) is not None and r[m]['truncated'] for r in rows)
                                 for m in ('qwen','apertus')}}
    false=[p for p in pairs if p['kind']=='false_premise']
    return {'planned_matched_records':108,'available_matched_records':sum(all(p.get(m) is not None for m in ('qwen','apertus')) for p in pairs),
        'overall':group_summary(pairs),
        'by_condition':{c:group_summary([p for p in pairs if p['context_condition']==c]) for c in ('bare','dates','attributes')},
        'by_base_pair':{b:{c:group_summary([p for p in pairs if p['base_pair_id']==b and p['context_condition']==c])
            for c in ('bare','dates','attributes')} for b in sorted({p['base_pair_id'] for p in pairs})},
        'breakdowns':{field:{value:group_summary([p for p in false if p[field]==value])
            for value in sorted({p[field] for p in false})} for field in ('variant','asserted_relation','direction_state')},
        'inference_or_semantic_rescoring_performed':False,
        'note':'Paired descriptive counts from preserved AI-assisted labels. Six correlated entity blocks; no independent-trial inference or model ranking.'}
