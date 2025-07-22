(* Context Amplification Validation *)
(* Tests: CONVEYANCE = BaseConveyance × Context^α × Physical_Grounding_Factor *)

(* Define the context amplification function *)
ContextAmplification[baseConveyance_, context_, alpha_, groundingFactor_] := 
  baseConveyance * context^alpha * groundingFactor

(* Test different alpha values across domains *)
alphaTests = {
  (* {Domain, Alpha, Description} *)
  {"Mathematics", 1.5, "Moderate amplification"},
  {"Physics", 1.8, "Strong amplification"},
  {"Philosophy", 2.0, "Maximum theoretical amplification"},
  {"Engineering", 1.6, "Practical amplification"}
};

(* Context values to test *)
contextValues = Range[0, 1, 0.1];

(* Validate bounded output for each alpha *)
Print["=== CONTEXT AMPLIFICATION VALIDATION ==="];
Print["Testing Context^α amplification across domains"];
Print[""];

boundednessResults = Table[
  {domain, alpha, description} = alphaTest;
  amplified = contextValues^alpha;
  minVal = Min[amplified];
  maxVal = Max[amplified];
  bounded = And[minVal >= 0, maxVal <= 1];
  
  {domain, alpha, 
   NumberForm[minVal, {3, 2}], 
   NumberForm[maxVal, {3, 2}], 
   If[bounded, "✓ Bounded", "✗ Unbounded"]},
  {alphaTest, alphaTests}
];

Grid[
  Prepend[boundednessResults, 
    {"Domain", "α", "Min", "Max", "Validation"}],
  Frame -> All,
  Background -> {None, {LightGreen, None}}
]

(* Plot amplification curves *)
Print[""];
Print["Amplification Curves:"];
Plot[
  Evaluate[Table[context^alpha, {alpha, {1.5, 1.8, 2.0}}]],
  {context, 0, 1},
  PlotLegends -> {"α = 1.5", "α = 1.8", "α = 2.0"},
  PlotLabel -> "Context Amplification: Context^α",
  AxesLabel -> {"Context Score", "Amplified Value"},
  PlotStyle -> {Red, Blue, Green},
  GridLines -> Automatic,
  ImageSize -> 400
]

(* Test complete conveyance calculation *)
Print[""];
Print["=== COMPLETE CONVEYANCE CALCULATION ==="];

conveyanceTests = {
  (* {Description, BaseConv, Context, Alpha, Grounding} *)
  {"Pure Theory (Foucault)", 0.5, 0.9, 1.5, 0.1},
  {"Applied Theory (PageRank)", 0.5, 0.9, 1.5, 0.8},
  {"Pure Code", 0.5, 0.3, 1.5, 0.9},
  {"Documentation", 0.5, 0.6, 1.5, 0.5}
};

conveyanceResults = Table[
  {desc, base, context, alpha, grounding} = test;
  conveyance = ContextAmplification[base, context, alpha, grounding];
  entropy = 1 - grounding;
  actionable = conveyance * (1 - entropy);
  
  {desc, 
   NumberForm[conveyance, {3, 2}],
   NumberForm[entropy, {3, 2}],
   NumberForm[actionable, {3, 2}]},
  {test, conveyanceTests}
];

Grid[
  Prepend[conveyanceResults, 
    {"Case", "Conveyance", "Entropy", "Actionable"}],
  Frame -> All,
  Background -> {None, {LightCyan, None}}
]

(* Validation summary *)
Print[""];
allBounded = And @@ (Last[#] == "✓ Bounded" & /@ boundednessResults);
If[allBounded,
  Print["✓ ALL TESTS PASSED: Context amplification remains bounded for α ∈ [1.5, 2.0]"],
  Print["✗ VALIDATION FAILED: Some α values produce unbounded results"]
]