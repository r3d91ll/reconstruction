(* Validation using real graph data from ArangoDB *)

(* Load the exported data *)
Get["/home/todd/reconstructionism/validation/wolfram/data/graph_data.wl"];

Print["================================================"];
Print["VALIDATION WITH REAL GRAPH DATA"];
Print["================================================"];
Print[""];

Print["Dataset Overview:"];
Print["  Papers: ", graphMetadata["paperCount"]];
Print["  Edges: ", graphMetadata["edgeCount"]];
Print["  Avg Context: ", graphMetadata["avgContext"]];
Print["  Std Context: ", graphMetadata["stdContext"]];
Print[""];

(* Test 1: Context Distribution Analysis *)
Print["TEST 1: CONTEXT DISTRIBUTION"];
Print["----------------------------"];

If[Length[contextScores] > 0,
  Module[{histogram, stats},
    stats = {
      "Min" -> Min[contextScores],
      "Max" -> Max[contextScores],
      "Mean" -> Mean[contextScores],
      "Median" -> Median[contextScores],
      "Skewness" -> Skewness[contextScores],
      "Kurtosis" -> Kurtosis[contextScores]
    };
    
    Print["Statistical Summary:"];
    Do[Print["  ", First[stat], ": ", N[Last[stat], 3]], {stat, stats}];
    
    (* Test for normal distribution *)
    normalTest = DistributionFitTest[contextScores, NormalDistribution[]];
    Print["Normal distribution test p-value: ", N[normalTest, 3]];
    Print[If[normalTest > 0.05, "✓ Consistent with normal distribution", 
             "✗ Not normally distributed"]];
  ],
  Print["No context scores available"];
];

Print[""];

(* Test 2: Context Amplification Validation *)
Print["TEST 2: CONTEXT^α AMPLIFICATION"];
Print["-------------------------------"];

If[Length[contextScores] > 0 && Length[amplifiedScores] > 0,
  Module[{alpha, predicted, mse, r2},
    (* Estimate alpha from data *)
    alpha = 1.5; (* Theoretical value *)
    
    (* Compare theoretical vs actual *)
    predicted = contextScores^alpha;
    mse = Mean[(amplifiedScores - predicted)^2];
    r2 = 1 - Total[(amplifiedScores - predicted)^2]/Total[(amplifiedScores - Mean[amplifiedScores])^2];
    
    Print["Alpha = ", alpha];
    Print["Mean Squared Error: ", N[mse, 6]];
    Print["R² Score: ", N[r2, 3]];
    
    (* Verify bounds *)
    bounded = And[Min[amplifiedScores] >= 0, Max[amplifiedScores] <= 1];
    Print[If[bounded, "✓ All amplified scores within [0,1]", "✗ Scores outside bounds"]];
    
    (* Check monotonicity *)
    paired = Transpose[{contextScores, amplifiedScores}];
    monotonic = And @@ (If[#1[[1]] < #2[[1]], #1[[2]] <= #2[[2]], True]& @@@ 
                       Subsets[paired, {2}]);
    Print[If[monotonic, "✓ Monotonic amplification", "✗ Non-monotonic behavior detected"]];
  ],
  Print["Insufficient data for amplification analysis"];
];

Print[""];

(* Test 3: Zero Propagation in Real Data *)
Print["TEST 3: ZERO PROPAGATION VERIFICATION"];
Print["-------------------------------------"];

(* Load the full JSON data for zero propagation test *)
jsonData = Import["/home/todd/reconstructionism/validation/wolfram/data/graph_data.json", "JSON"];

If[KeyExistsQ[jsonData, "zero_propagation"],
  Module[{zpData, violations},
    zpData = jsonData["zero_propagation"];
    violations = 0;
    
    Do[
      Module[{info, hasZero, expectZero},
        info = item["INFORMATION"];
        hasZero = Or @@ (# == 0 & /@ {item["WHERE"], item["WHAT"], 
                                      item["CONVEYANCE"], item["TIME"]});
        expectZero = hasZero && info != 0;
        
        If[expectZero,
          violations++;
          Print["✗ Violation: ", item["title"], " has zero dimension but info=", info];
        ];
      ],
      {item, zpData}
    ];
    
    Print["Tested ", Length[zpData], " papers"];
    Print[If[violations == 0, 
             "✓ Zero propagation holds for all test cases",
             "✗ Found " <> ToString[violations] <> " violations"]];
  ],
  Print["No zero propagation data available"];
];

Print[""];

(* Test 4: Theory-Practice Bridge Detection *)
Print["TEST 4: BRIDGE CANDIDATE ANALYSIS"];
Print["---------------------------------"];

If[KeyExistsQ[jsonData, "bridge_candidates"],
  Module[{bridges, highNorm, longAbstract, multiCategory},
    bridges = jsonData["bridge_candidates"];
    
    (* Identify potential bridges based on multiple signals *)
    highNorm = Select[bridges, #["embedding_norm"] > 5 &];
    longAbstract = Select[bridges, #["abstract_length"] > 1000 &];
    multiCategory = Select[bridges, #["category_count"] > 2 &];
    
    Print["Potential bridge indicators:"];
    Print["  High embedding norm: ", Length[highNorm], " papers"];
    Print["  Long abstracts: ", Length[longAbstract], " papers"];
    Print["  Multiple categories: ", Length[multiCategory], " papers"];
    
    (* Find papers meeting multiple criteria *)
    bridgeScore[paper_] := Count[{
      paper["embedding_norm"] > 5,
      paper["abstract_length"] > 1000,
      paper["category_count"] > 2
    }, True];
    
    topBridges = TakeLargestBy[bridges, bridgeScore, 3];
    Print["\nTop bridge candidates:"];
    Do[
      Print["  ", i, ". ", paper["title"], " (score=", bridgeScore[paper], ")"],
      {i, Length[topBridges]}, {paper, topBridges}
    ];
  ],
  Print["No bridge candidate data available"];
];

Print[""];
Print["================================================"];
Print["VALIDATION COMPLETE"];
Print["================================================"];