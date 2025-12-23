import random

def generate_period_search_in(filename="period_search_in"):
    with open(filename, "w") as f:
        # 1. Period Range
        # per_start, per_step_coef, per_end, ia_prd
        f.write("2.0 1.0 10.0 1\n")
        
        # 2. Epoch JD0
        f.write("2451545.0\n")
        
        # 3. Phi0
        f.write("0.0\n")
        
        # 4. Convexity weight
        f.write("0.1\n")
        
        # 5. Lmax Mmax
        f.write("2 2\n")
        
        # 6. nrows (triangulation)
        f.write("4\n")
        
        # 7. Initial guess (par1, ia1) - e.g. Minnaert params
        f.write("0.5 1\n")
        f.write("0.5 1\n")
        f.write("0.5 1\n")
        
        # 8. Lambert coeff
        f.write("0.1 1\n")
        
        # 9. Stop condition
        f.write("10.0\n")
        
        # 10. Min iterations
        f.write("5\n")
        
        # 11. Alamda incr
        f.write("10.0\n")
        
        # 12. Alamda start
        f.write("0.001\n")
        
        # 13. Lcurves
        num_curves = 1
        f.write(f"{num_curves}\n")
        
        # Lightcurve 1
        num_points = 10
        is_rel = 1 # 1-itemp=0 -> itemp=1 -> Inrel=0. 1-0 = 1.
        # Code: fscanf(infile, "%d %d", &Lpoints[i], &i_temp); Inrel[i] = 1 - i_temp;
        # So providing '1' means Inrel=0 ? No. 1-1=0.
        # Let's say we want Inrel=1 (Relative). So itemp=0.
        f.write(f"{num_points} 0\n") 
        
        for i in range(num_points):
            jd = 2451545.0 + i * 0.1
            brightness = 1.0 + random.random() * 0.1
            # Sun (x,y,z) Earth (x,y,z)
            sun = "1.0 0.0 0.0"
            earth = "0.9 0.1 0.0"
            f.write(f"{jd} {brightness}\n")
            f.write(f"{sun}\n")
            f.write(f"{earth}\n")
            
        # Weights
        # Code: while (fscanf... lc_number, weight)
        f.write(f"1 1.0\n")

if __name__ == "__main__":
    generate_period_search_in()
    print("Generated period_search_in")
