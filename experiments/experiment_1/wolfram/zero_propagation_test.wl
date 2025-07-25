(* Zero Propagation Test for Information Reconstructionism *)
(* Tests the core principle: If ANY dimension = 0, then Information = 0 *)

(* Define the core equation *)
InformationValue[where_, what_, conveyance_, time_, frame_] := 
  where * what * conveyance * time * frame

(* Test cases demonstrating zero propagation *)
testCases = {
  (* {WHERE, WHAT, CONVEYANCE, TIME, FRAME, Description} *)
  {1.0, 1.0, 1.0, 1.0, 1.0, "All dimensions present"},
  {0.0, 1.0, 1.0, 1.0, 1.0, "WHERE = 0 (no location)"},
  {1.0, 0.0, 1.0, 1.0, 1.0, "WHAT = 0 (no content)"},
  {1.0, 1.0, 0.0, 1.0, 1.0, "CONVEYANCE = 0 (no actionability)"},
  {1.0, 1.0, 1.0, 0.0, 1.0, "TIME = 0 (no temporal context)"},
  {1.0, 1.0, 1.0, 1.0, 0.0, "FRAME = 0 (observer cannot perceive)"},
  {0.5, 0.8, 0.9, 0.7, 1.0, "All partial values"},
  {0.1, 0.1, 0.1, 0.1, 0.1, "All minimal values"},
  {0.0, 0.5, 0.8, 0.9, 1.0, "Single zero propagates"}
};

(* Calculate results *)
results = Table[
  {where, what, conveyance, time, frame, description} = testCase;
  info = InformationValue[where, what, conveyance, time, frame];
  hasZero = Or[where == 0, what == 0, conveyance == 0, time == 0, frame == 0];
  {
    description,
    NumberForm[info, {5, 4}],
    If[hasZero && info == 0, "✓ Zero propagated", 
       If[!hasZero && info > 0, "✓ Non-zero preserved", "✗ FAILED"]]
  },
  {testCase, testCases}
];

(* Display results *)
Print["=== ZERO PROPAGATION VALIDATION ==="];
Print["Core Principle: Information(i→j|S-O) = WHERE × WHAT × CONVEYANCE × TIME × FRAME"];
Print["If ANY dimension = 0, then Information = 0"];
Print[""];

Grid[
  Prepend[results, {"Test Case", "Information Value", "Validation"}],
  Frame -> All,
  Background -> {None, {LightBlue, None}},
  Spacings -> {1, 0.5}
]

(* Summary *)
Print[""];
failedTests = Count[results, {_, _, "✗ FAILED"}];
If[failedTests == 0,
  Print["✓ ALL TESTS PASSED: Zero propagation principle validated"],
  Print["✗ VALIDATION FAILED: ", failedTests, " tests failed"]
]