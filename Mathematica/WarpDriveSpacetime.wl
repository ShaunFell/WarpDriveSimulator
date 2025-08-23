(* ::Package:: *)

(* ::Input:: *)
(**)


(*Download OGRE if not present, then include*)

Module[{depPath},
  depPath = FileNameJoin[{$UserBaseDirectory, "Applications", "OGRe.m"}];
  
  If[!FileExistsQ[depPath],
    (* Download the dependency if missing *)
    Print["Downloading OGRe..."];
    result = Quiet@Check[
    URLDownload[
      "https://raw.githubusercontent.com/bshoshany/OGRe/master/OGRe.m",
      depPath
    ],
    $Failed
    ];
     
    If[result == $Failed,
    MessageDialog["Could not download OGRe package."];
    Abort[],
    "OGRe downloaded successfully"
    ];
  ];
  
  (* Silently load the dependency *)
  Block[{Print, OGRePrint, CellPrint},
    Get[depPath]
  ];
];


(*Warp Drive Spacetime Framework for Mathematica using OGRe*)
BeginPackage["WarpDriveSpacetime`", {"OGRe`"}];


DefineSpacetime::usage = "DefineSpacetime[name, coords, metricComponents] creates a spacetime using OGRe.";
MinkowskiSpacetime::usage = "MinkowskiSpacetime[] creates Minkowski spacetime in Cartesian coordinates.";
AlcubierreSpacetime::usage = "AlcubierreSpacetime[velocity, radius, thickness] creates Alcubierre warp drive spacetime.";
SpacetimeMetric::usage = "SpacetimeMetric[spacetime] returns the metric tensor ID.";
SpacetimeChristoffel::usage = "SpacetimeChristoffel[spacetime] computes and returns Christoffel symbols.";
SpacetimeRiemann::usage = "SpacetimeRiemann[spacetime] computes and returns Riemann tensor.";
SpacetimeRicci::usage = "SpacetimeRicci[spacetime] computes and returns Ricci tensor and scalar.";
SpacetimeEinstein::usage = "SpacetimeEinstein[spacetime] computes and returns Einstein tensor.";
WarpFunction::usage = "WarpFunction[r, R, \[Sigma]] computes the Alcubierre warp function.";
SpacetimeVisualization::usage = "SpacetimeVisualization[spacetime] creates visualization.";
SpacetimeAnalysis::usage = "SpacetimeAnalysis[spacetime] analyzes the spacetime.";
GeodesicEquations::usage = "GeodesicEquations[spacetime] generates geodesic equations.";
NullCondition::usage = "NullCondition[spacetime, velocity] checks null geodesic condition.";
GetSpacetimeComponents::usage = "GetSpacetimeComponents[spacetime] gets metric components.";
ExportSpacetime::usage = "ExportSpacetime[spacetime] exports spacetime data.";
ShowMetric::usage = "ShowMetric[metric] "



Begin["`Private`"];


(* =============================================================================GENERAL SPACETIME FRAMEWORK=============================================================================*)

(*Define a spacetime as an Association with all relevant information*)
Options[DefineSpacetime] = {
Type->"General"
}
DefineSpacetime[name_String, coords_List, metricComponents_List, opts: OptionsPattern[]] := Module[{
  coordsID, metricID, spacetime
  },
  
  (* Create coordinate system using actual OGRe syntax *)
  coordsID = name <> "Coords";
  TNewCoordinates[coordsID, coords];
  
  (* Create metric using actual OGRe syntax *)
  metricID = name <> "Metric";
  TNewMetric[metricID, coordsID, metricComponents, "g"];
  
  (* Return spacetime as Association *)
  spacetime = <|
    "Name" -> name,
    "CoordinatesID" -> coordsID,
    "MetricID" -> metricID,
    "Coordinates" -> coords,
    "Type" -> OptionValue[Type]
  |>;
  
  spacetime
]

(* =============================================================================MINKOWSKI SPACETIME=============================================================================*)

MinkowskiSpacetime[] := Module[{coords, metric},
  Global`coords = {t, x, y, z};
  
  (* Minkowski metric in (-,+,+,+) signature *)
  metric = {
    {-1, 0, 0, 0},
    {0, 1, 0, 0},
    {0, 0, 1, 0},
    {0, 0, 0, 1}
  };
  
  DefineSpacetime["Minkowski", coords, metric, Type->"Flat"] 
]

(* =============================================================================ALCUBIERRE WARP DRIVE SPACETIME=============================================================================*)

(* Alcubierre warp function with smooth cutoff *)
WarpFunction[r_, R_, \[Sigma]_: 0.5] := 
  (Tanh[\[Sigma] (r + R)] - Tanh[\[Sigma] (r - R)])/(2 Tanh[\[Sigma] R])

(* Alcubierre spacetime *)
AlcubierreSpacetime[vs_: 1, R_: 10, \[Sigma]_: 0.5] := Module[{
  coords, metric, rs, warpFunc, xs, spacetime
  },
  
  coords = {t, x, y, z};
  
  (* Bubble center trajectory: xs = vs*t *)
  xs = vs*t;
  
  (* Distance from bubble center *)
  rs = Sqrt[(x - xs)^2 + y^2 + z^2];
  
  (* Warp function *)
  warpFunc = WarpFunction[rs, R, \[Sigma]];
  
  (* Alcubierre metric in standard form *)
  (* ds\.b2 = -dt\.b2 + (dx - vs*f(rs)*dt)\.b2 + dy\.b2 + dz\.b2 *)
  metric = {
    {-1 + vs^2*warpFunc^2, -vs*warpFunc, 0, 0},
    {-vs*warpFunc, 1, 0, 0},
    {0, 0, 1, 0},
    {0, 0, 0, 1}
  };
  
  spacetime = DefineSpacetime["Alcubierre", coords, metric, Type->"WarpDrive"];
  spacetime["Velocity"] = vs;
  spacetime["BubbleRadius"] = R;
  spacetime["Thickness"] = \[Sigma];
  spacetime["WarpFunction"] = warpFunc;
  spacetime["BubbleTrajectory"] = xs;
  
  spacetime
]

(* =============================================================================SPACETIME ANALYSIS FUNCTIONS=============================================================================*)

(* Get metric tensor ID *)
SpacetimeMetric[spacetime_Association] := 
  spacetime["MetricID"]

(* Get metric components using OGRe *)
GetSpacetimeComponents[spacetime_Association] := 
  TGetComponents[spacetime["MetricID"]]

(* Compute Christoffel symbols using OGRe *)
SpacetimeChristoffel[spacetime_Association] := Module[{christoffelID},
  christoffelID = TCalcChristoffel[spacetime["MetricID"]];
  christoffelID
]

(* Compute Riemann tensor using OGRe *)
SpacetimeRiemann[spacetime_Association] := Module[{riemannID},
  riemannID = TCalcRiemannTensor[spacetime["MetricID"]];
  riemannID
]

(* Compute Ricci tensor and scalar using OGRe *)
SpacetimeRicci[spacetime_Association] := Module[{ricciTensorID, ricciScalarID},
  ricciTensorID = TCalcRicciTensor[spacetime["MetricID"]];
  ricciScalarID = TCalcRicciScalar[spacetime["MetricID"]];
  <|"RicciTensor" -> ricciTensorID, "RicciScalar" -> ricciScalarID|>
]

(* Compute Einstein tensor using OGRe *)
SpacetimeEinstein[spacetime_Association] := Module[{einsteinID},
  einsteinID = TCalcEinsteinTensor[spacetime["MetricID"]];
  einsteinID
]

(* =============================================================================GEODESIC AND LIGHT RAY FUNCTIONS=============================================================================*)

GeodesicEquations[spacetime_Association] := Module[{geodesicID},
  geodesicID = TCalcGeodesicFromChristoffel[spacetime["MetricID"], spacetime["CoordinatesID"]];
  geodesicID
]

(* Null condition for light rays *)
NullCondition[spacetime_Association, velocity_List] := Module[{
  g, metricComponents, nullCond
  },
  metricComponents = TGetComponents[spacetime["MetricID"]];
  nullCond = Sum[
    metricComponents[[\[Mu], \[Nu]]] * velocity[[\[Mu]]] * velocity[[\[Nu]]], 
    {\[Mu], 1, 4}, {\[Nu], 1, 4}
  ];
  nullCond == 0 // Simplify
]


(* =============================================================================VISUALIZATION AND ANALYSIS=============================================================================*)

(* Spacetime visualization *)
SpacetimeVisualization[spacetime_Association] := Module[{
  coords, params
  },
  coords = spacetime["Coordinates"];
  
  Switch[spacetime["Type"],
    "Flat",
    (* Minkowski light cone *)
    Plot3D[Sqrt[x^2 + y^2], {x, -10, 10}, {y, -10, 10}, 
      PlotStyle -> Opacity[0.7], 
      PlotLabel -> "Minkowski Light Cone",
      AxesLabel -> {"x", "y", "t"},
      ImageSize -> Medium
    ],
    
    "WarpDrive",
    (* Alcubierre warp field visualization *)
    With[{vs = spacetime["Velocity"], R = spacetime["BubbleRadius"], 
          \[Sigma] = spacetime["Thickness"]},
      Manipulate[
        Plot3D[WarpFunction[Sqrt[(x - vs*time)^2 + y^2], R, \[Sigma]], 
          {x, -20, 20}, {y, -10, 10},
          PlotRange -> {-1, 1},
          PlotLabel -> "Alcubierre Warp Field at t = " <> ToString[time],
          AxesLabel -> {"x", "y", "f(r)"},
          ColorFunction -> "RainbowOpacity",
          ImageSize -> Medium
        ],
        {time, 0, 10, 0.1}
      ]
    ],
    
    _,
    Graphics[Text["Visualization not implemented for this spacetime type"], 
      ImageSize -> Medium]
  ]
]

(* Comprehensive spacetime analysis *)
SpacetimeAnalysis[spacetime_Association] := Module[{
  analysis, metricComponents
  },
  
  metricComponents = TGetComponents[spacetime["MetricID"]];
  
  analysis = <|
    "Spacetime" -> spacetime["Name"],
    "Coordinates" -> spacetime["Coordinates"],
    "MetricDeterminant" -> Det[metricComponents] // Simplify,
    "Type" -> spacetime["Type"]
  |>;
  
  (* Add type-specific analysis *)
  Switch[spacetime["Type"],
    "Flat",
    analysis["CurvatureType"] = "Flat",
    
    "WarpDrive",
    analysis["WarpVelocity"] = spacetime["Velocity"];
    analysis["BubbleRadius"] = spacetime["BubbleRadius"];
    analysis["CurvatureType"] = "Curved",
    
    _,
    analysis["CurvatureType"] = "Unknown"
  ];
  
  analysis
]

(* =============================================================================UTILITY FUNCTIONS=============================================================================*)

(* Display line element using OGRe *)
ShowLineElement[spacetime_Association] := 
  TLineElement[spacetime["MetricID"]]

(* Show metric components using OGRe *)
ShowMetric[spacetime_Association] := 
  TShow[spacetime["MetricID"]]

(* List non-zero components using OGRe *)
ListMetric[spacetime_Association] := 
  TList[spacetime["MetricID"]]

(* Export spacetime information *)
ExportSpacetime[spacetime_Association] := 
  <|
    "SpacetimeInfo" -> spacetime,
    "MetricComponents" -> TGetComponents[spacetime["MetricID"]],
    "OGReData" -> TExport[spacetime["MetricID"]]
  |>

(* =============================================================================
   CONVENIENCE FUNCTIONS FOR ANALYSIS
   ============================================================================= *)

(* Calculate and display curvature tensors *)
AnalyzeCurvature[spacetime_Association] := Module[{
  christoffelID, riemannID, ricciData, einsteinID
  },
  
  Print["Computing Christoffel symbols..."];
  christoffelID = SpacetimeChristoffel[spacetime];
  
  Print["Computing Riemann tensor..."];
  riemannID = SpacetimeRiemann[spacetime];
  
  Print["Computing Ricci tensor and scalar..."];
  ricciData = SpacetimeRicci[spacetime];
  
  Print["Computing Einstein tensor..."];
  einsteinID = SpacetimeEinstein[spacetime];
  
  Print["Analysis complete. Use TShow[tensorID] to display components."];
  
  <|
    "Christoffel" -> christoffelID,
    "Riemann" -> riemannID,
    "Ricci" -> ricciData,
    "Einstein" -> einsteinID
  |>
]

(* Calculate geodesics *)
AnalyzeGeodesics[spacetime_Association] := Module[{geodesicID},
  Print["Computing geodesic equations..."];
  geodesicID = GeodesicEquations[spacetime];
  Print["Geodesic equations computed. Use TShow[\"", geodesicID, "\"] to display."];
  geodesicID
]

End[];
EndPackage[];


