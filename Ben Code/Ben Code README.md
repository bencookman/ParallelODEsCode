This is the README for my (Benjamin Cookman's) project IV code, titled:

'Arbitrary Order Parallel and Sequential Time Integrators'.

========================================

INTRO

This project is, at least loosely speaking, about how integral deferred correction (IDC) methods work and how they build up to form parallel methods - most notably the revisionist integral deferred correction (RIDC) method of Christlieb et al.. In reality, the dissertation focussed primarily (~30 pages or so) on the IDC method and variations thereof. Then, it focuses on how this is modified to form a 'new' (at least to the best of my knowledge) parallel formulation of IDC, dubbed parallel composite IDC (PCIDC, or PCSDC when special nodes are used within each composition).

The contents of this folder,

ParallelODEsCode / Ben Code,

as well as this readme file assume you have read my dissertation, otherwise it'd be frankly hopeless to explain each algorithm on the fly as they are defined. With this in mind, this brings us to the sections of this file:

I		- How To Use This Code + Code Structure
II  	- Notation / Nomenclature Used
III 	- ODE Systems
IV		- Runge-Kutta Methods
V		- Integration Matrices + Tests
VI		- The Sequential Algorithms
VII		- The 'Parallel' Algorithms
VIII	- Convergence Tests
IX		- Stability Tests

----------------------------------------

I . How To Use This Code + Code Structure 

....................

Code Structure

As mentioned above, all of the code for this project and that which is used for the dissertation is held in this '.../ParallelODEsCode/Ben Code' folder. Sepcifically, the code is held in the '.../ParallelODEsCode/Ben Code/src' folder. Output figures are held in the '.../ParallelODEsCode/Ben Code/output' folder, although most of the contents have been deleted (most of it was horrible looking plots noone ought to be subjected to).

Here's a brief rundown on what all the Julia .jl files do:

'ProjectTools.jl' - This is the main source code folder, containing the ProjectTools module which contains structures for testing, integration matrices and all the relevant algorithms (IDC and RK).

'IDC_test.jl' - This contains all of the convergence tests for this project. Initially used just to test whether or not an algorithm converged with the correct order of accuracy, it later was used to generate the high-quality plots seen in the dissertation.

'stability.jl' - This contains the numerical stability region calculators which, given a ODE solver algorithm (within '.../ParallelODEsCode/Ben Code') wil give back a stability region.

'integration matrix test.jl' - Contains accuracy tests for integration matrices to expose issues of the Runge phenomenon and catastrophic cancellation in these matrices.

'implicit_methods.jl' - File containing implicit methods and newton root-finding algorithm. These did not end up being used in the dissertation beyond a brief overview.

'heat_equation.jl' - uses forward Euler method coupled with the FFT to numerically solve the heat equation. Initially, I had planned on studying PDEs further using the discrete Fourier transform and RIDC, but this did not end up being where the project went.

'vorticity_equation_2D_FE.jl' - Another example of numerical PDEs stuff which was not delved too deeply into in the dissertation.

'KdV_equation.jl' - Ditto. The soliton solutions to the KdV equation are well studied, so would be an interesting, highly visual example of a PDE to solve utilising RIDC. Once again, base IDC and PCIDC methods were focussed on much more in the dissertation in the end.

You'll also find within '.../ParallelODEsCode/Ben Code' the files 'Manifest.toml' and 'Project.toml', which are Julia environment files and allow for the versions of packages used in a julia environment to be coordinated (see https://pkgdocs.julialang.org/v1/toml-files/).

***JULIA USES INDEXING FROM 1. THIS IS ANNOYING.*** i got really good at converting from 0 to 1 indexing and back though (silver linings).

....................

How To Use This Code

This is probably the most important part of this readme as it means readers may actually be able to use this code. in some ways, Julia is weird and I only learned how to code in it in August 2022 so wasn't entirely sure how to structure to code from the get-go (I'm not a software engineer sorry). This has caused a somewhat frustrating usability that should be avoided in future projects

Anyway, without further ado, how is this code actually use??

First of all, you must move to the cloned 'probably GitHub/ParallelODEsCode' folder and then enter the Julia REPL (***I HAVE USED JULIA VERSION 1.7.3, YOU SHOULD TO***) with:

dir> julia --project="Ben Code"

where the tag '--project="Ben Code"' enters the 'Ben Code' environment (this is what the 'Project.toml' and 'Manifest.toml' files are for), so your REPL will guarantee it uses the correct versions of packages used by code (so there is no clashing). You'll see something like this:
               _
   _       _ _(_)_     |  Documentation: https://docs.julialang.org
  (_)     | (_) (_)    |
   _ _   _| |_  __ _   |  Type "?" for help, "]?" for Pkg help.
  | | | | | | |/ _` |  |
  | | |_| | | | (_| |  |  Version 1.7.3 (2022-05-06)
 _/ |\__'_|_|_|\__'_|  |  Official https://julialang.org/ release
|__/                   |

julia> 

From now on, you may choose to use other packages in the REPL, if you do you'll have to add them as well with

julia> ] add <Package Name>

As far as I'm aware, it should download the correct versions of these packages according to the 'Ben Code' environment which you should be in. You can check your current environment with

julia> ]

and it should turn into

(Ben Code) pkg> 

Now, I highly recommend you use the Revise.jl package:

julia> ] add Revise
julia> use Revise

so that you can modify code and not have to reimport the altered files everytime (so you only have to deal with one of the compile times). All of code used stems from the 'ProjectTools' module, contained in the 'ProjectTools.jl' file, so include this with:

julia> includet("Ben Code/src/ProjectTools.jl")

This is the same way you include any other file, where the 'includet' function is used instead of the usual 'include' to invoke the Revise package, so you won't need to recompile each file everytime a small change is made. Since this file just contains the ProjectTools module, if you want to use its contents, you'll have to do

julia> using .ProjectTools

which will expose all of the contents of the module which are 'export'ed. Generally, I use this file in conjunction with another which tests its contents. To do this, make sure you don't do the above (the 'using ...' bit) and do

julia> includet("Ben Code/src/whatever file.jl")

Now you can use any of the contents of 'whatever file.jl' (including the entire ProjectTools module, provided the lines

include("ProjectTools.jl")
using .ProjectTools

are included in 'whatever file.jl', which it is in basically every important file) as you'd like. By including the 'ProjectTools.jl' and then 'whatever file.jl' files in this order, the contents of the 'ProjectTools' module which is exposed in the julia REPL will update as the 'ProjectTools.jl' (not that you'll be changing that file much).

----------------------------------------

II . Notation / Nomenclature Used

The code in this folder of the GitHub repository was written before the dissertation was. As a result, different notation and nomenclature are used for variables in many cases here than in the dissertation. Here are the main differences:

Code notation | Dissertation notation

η (sometimes u) | y with tilde ~, the numerical approximation
k (when not in an RK stage) | i, general sequence index, e.g. y_k for k=1 to N+1 (reminder, indexing starts at 1 here)
integration matrix | generic weights matrix, (because these quadrature weights give us numerical integrals in a linear way)
S | W hat, the generic weights matrix
J | G, number of groups used in approximation

----------------------------------------

III . ODE Systems

* would like to separate the numerical parameters from IVP parameters

It is unwise to program an ODE solver from scratch. To test the algorithms relevent to my diss though, this is necessary to prove I fully understand each method and their variations. Everytime an ordinary differential equation is solved, the data on that IVP must be given to the numerical method as well as some number of numerical parameters. To separate the two, I've introduced the ODESystem structure to store the information on these IVPs to be passed to a numerical method I've coded.

This can be used with the ODETestSystem which additionally stores the exact solution as a function of time. This is useful for testing in general but convergence tests especially.

----------------------------------------

IV . Runge-Kutta Methods

This bit's actually quite cool (at least to me)!

So, as mentioned in the dissertation, any RK method can be expressed as the Butcher tableau:

c | a
--+--  .
  | b
 
So to store a whole RK method, in theory we only need to store these values. This is done in the RKMethod structure with the extra field 'orderofaccuracy' (this is just useful information that will be readily available whenever an RKMethod is introduced).

How might we use these structures as actual numerical algorithms though? Well, we can take an IVP (which will be an ODESystem struct) and RKMethod as input along with some number of time steps N and perform each step of the RK method using its tableau.

The really cool part of it is actually this: the line:

(RK_method::RKMethod)(ODE_system::ODESystem, N) = RK_general(ODE_system, N; RK_method = RK_method)

means that any instance of an RKMethod, say 'RK1_forward_euler' for the first order FE method, can be called as though it were a function which takes an IVP 'ODE_system' as input along with the resolution N. This matches the way other numerical methods I have programmed, e.g. 'IDC-FE', are coded and so can be used almost intechangeably with very little difference in how they are used. The only difference is the numerical parameters they require which i couldnt be bothered to factor out into their own numerical parameter structure as well.

So if you want a fourth order Runge-Kutta approximation to a generic IVP 'my_ivp' with 100 time steps, you would do:

(t_values, y_values) = RK4_standard(my_ivp, 100)

Nice!

----------------------------------------

V . Integration Matrices + Tests

Integration matrices (or generic weights matrices as they are known in the dissertation) are a crucial part of any integral deferred correction method. In ProjectTools, we introduce several methods to calculate these matrices, arrays which store these matrices and functions to calculate Lagrange basis functions.

Before using any integration matrix, it is best that you calculate it and store it for future use (they can be quite expensive to calculate). This is done through:

fill_integration_matrix_array_uniform(<# quadrature nodes>) or fill_integration_matrix_array_lobatto() <# quadrature nodes>etc.

The separation between quadrature points and time steps when making the integration matrix is further detailed in the dissertation. Here it is important as it allows us to calculate any generic integration matrix.

----------------------------------------

VI . The Sequential Algorithms



----------------------------------------

VII . The 'Parallel' Algorithms



----------------------------------------

VIII . Convergence Tests



----------------------------------------

IX . Stability Tests









