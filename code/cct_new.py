import numpy as np
import pandas as pd
import pymc as pm
import arviz as az
import matplotlib.pyplot as plt
import seaborn as sns

def load_data(filepath):
    """
    Load the plant knowledge dataset and return it as a numpy array,
    excluding the Informant ID column.
    
    Parameters:
    -----------
    filepath : str
        Path to the CSV file containing the plant knowledge data
        
    Returns:
    --------
    np.ndarray
        2D array of binary responses where rows are informants and columns are questions
    """
    # Load data from CSV
    data = pd.read_csv(filepath)
    
    # Extract informant IDs for later reference
    informant_ids = data['Informant_ID'].values
    
    # Remove the Informant_ID column and convert to numpy array
    response_data = data.drop(columns=['Informant_ID']).values
    
    return response_data, informant_ids

def build_cct_model(X):
    """
    Build a Cultural Consensus Theory model using PyMC.
    
    Parameters:
    -----------
    X : np.ndarray
        2D array of binary responses where rows are informants and columns are questions
        
    Returns:
    --------
    pm.Model
        PyMC model for Cultural Consensus Theory
    """
    N, M = X.shape  # N = number of informants, M = number of questions
    
    with pm.Model() as cct_model:
        # Prior for informant competence (D)
        # Using Beta distribution with parameters that favor competence above 0.5
        # Beta(2,1) puts more weight on values above 0.5 while still allowing for the full range
        D_raw = pm.Beta('D_raw', alpha=2, beta=1, shape=N)
        
        # Transform to constrain D between 0.5 and 1
        D = pm.Deterministic('D', 0.5 + 0.5 * D_raw)
        
        # Prior for consensus answers (Z)
        # Bernoulli(0.5) reflects no prior knowledge about the correct answers
        Z = pm.Bernoulli('Z', p=0.5, shape=M)
        
        # Reshape D for broadcasting
        D_reshaped = D[:, None]  # Shape: (N, 1)
        
        # Calculate response probability using the CCT formula
        # p_ij = Z_j * D_i + (1 - Z_j) * (1 - D_i)
        p = Z * D_reshaped + (1 - Z) * (1 - D_reshaped)
        
        # Define the likelihood
        X_obs = pm.Bernoulli('X_obs', p=p, observed=X)
        
    return cct_model

def run_inference(model, draws=2000, chains=4, tune=1000):
    """
    Run MCMC sampling to perform inference on the model.
    
    Parameters:
    -----------
    model : pm.Model
        PyMC model to sample from
    draws : int
        Number of samples to draw
    chains : int
        Number of chains to run
    tune : int
        Number of tuning steps
        
    Returns:
    --------
    az.InferenceData
        Trace object containing the samples
    """
    with model:
        # Use the NUTS sampler for efficient exploration of the posterior
        trace = pm.sample(draws=draws, chains=chains, tune=tune, random_seed=42)
    
    return trace

def analyze_competence(trace, informant_ids=None):
    """
    Analyze the competence estimates for each informant.
    
    Parameters:
    -----------
    trace : az.InferenceData
        MCMC samples from the model
    informant_ids : np.ndarray, optional
        Array of informant IDs
        
    Returns:
    --------
    pd.DataFrame
        DataFrame with competence estimates for each informant
    """
    # Extract competence samples
    D_samples = az.extract(trace, var_names=['D']).D.values
    
    # Calculate mean competence for each informant
    mean_competence = D_samples.mean(axis=0)
    
    # Create competence DataFrame
    if informant_ids is not None:
        competence_df = pd.DataFrame({
            'Informant_ID': informant_ids,
            'Competence': mean_competence
        })
    else:
        competence_df = pd.DataFrame({
            'Informant_ID': [f"Informant_{i+1}" for i in range(len(mean_competence))],
            'Competence': mean_competence
        })
    
    # Sort by competence (highest to lowest)
    competence_df = competence_df.sort_values('Competence', ascending=False).reset_index(drop=True)
    
    return competence_df

def analyze_consensus(trace, X):
    """
    Analyze the consensus answers for each question.
    
    Parameters:
    -----------
    trace : az.InferenceData
        MCMC samples from the model
    X : np.ndarray
        The original data matrix
        
    Returns:
    --------
    pd.DataFrame
        DataFrame with consensus estimates and majority vote for each question
    """
    # Extract consensus samples
    Z_samples = az.extract(trace, var_names=['Z']).Z.values
    
    # Calculate mean consensus probability for each question
    mean_consensus_probs = Z_samples.mean(axis=0)
    
    # Determine most likely consensus answers (1 if p > 0.5, else 0)
    consensus_answers = (mean_consensus_probs > 0.5).astype(int)
    
    # Calculate majority vote answers
    majority_vote = (X.mean(axis=0) > 0.5).astype(int)
    
    # Create consensus DataFrame
    consensus_df = pd.DataFrame({
        'Question': [f"Q{i+1}" for i in range(len(mean_consensus_probs))],
        'Consensus_Probability': mean_consensus_probs,
        'Consensus_Answer': consensus_answers,
        'Majority_Vote': majority_vote,
        'Agreement': consensus_answers == majority_vote
    })
    
    return consensus_df

def visualize_results(trace, competence_df, consensus_df):
    """
    Visualize the results of the CCT analysis.
    
    Parameters:
    -----------
    trace : az.InferenceData
        MCMC samples from the model
    competence_df : pd.DataFrame
        DataFrame with competence estimates
    consensus_df : pd.DataFrame
        DataFrame with consensus estimates
    """
    # Plot competence estimates
    plt.figure(figsize=(10, 6))
    sns.barplot(x='Informant_ID', y='Competence', data=competence_df)
    plt.title('Estimated Informant Competence')
    plt.ylabel('Competence (probability of knowing correct answer)')
    plt.xticks(rotation=45)
    plt.axhline(y=0.5, color='r', linestyle='--', alpha=0.5, label='Chance level')
    plt.ylim(0.45, 1.0)
    plt.legend()
    plt.tight_layout()
    plt.savefig('competence_estimates.png')
    
    # Plot consensus answers vs majority vote
    plt.figure(figsize=(12, 6))
    
    x = np.arange(len(consensus_df))
    width = 0.35
    
    plt.bar(x - width/2, consensus_df['Consensus_Probability'], width, label='CCT Model Probability')
    plt.bar(x + width/2, consensus_df['Majority_Vote'], width, label='Majority Vote', alpha=0.7)
    
    plt.axhline(y=0.5, color='r', linestyle='--', alpha=0.3)
    plt.xlabel('Question')
    plt.ylabel('Probability/Vote')
    plt.title('Consensus Answers vs Majority Vote')
    plt.xticks(x, consensus_df['Question'])
    plt.ylim(0, 1.1)
    plt.legend()
    plt.tight_layout()
    plt.savefig('consensus_vs_majority.png')
    
    # Plot trace and posterior distributions
    az.plot_trace(trace, var_names=['D'])
    plt.savefig('trace_D.png')
    
    az.plot_trace(trace, var_names=['Z'])
    plt.savefig('trace_Z.png')
    
    # Assess convergence with pair plot
    az.plot_pair(trace, var_names=['D'], kind='scatter')
    plt.savefig('pair_plot_D.png')

def generate_report(trace, competence_df, consensus_df):
    """
    Generate a report summarizing the CCT analysis results.
    
    Parameters:
    -----------
    trace : az.InferenceData
        MCMC samples from the model
    competence_df : pd.DataFrame
        DataFrame with competence estimates
    consensus_df : pd.DataFrame
        DataFrame with consensus estimates
        
    Returns:
    --------
    str
        Report text
    """
    # Get summary statistics
    summary = az.summary(trace)
    
    # Check convergence
    converged = (summary['r_hat'] < 1.1).all()
    convergence_status = "Good convergence" if converged else "Poor convergence"
    
    # Get top and bottom informants
    top_informant = competence_df.iloc[0]
    bottom_informant = competence_df.iloc[-1]
    
    # Count disagreements between CCT and majority vote
    disagreements = (~consensus_df['Agreement']).sum()
    
    report = f"""
    # Cultural Consensus Theory Model Report
    
    ## Model Structure and Priors
    The CCT model was implemented using PyMC with the following structure:
    - Informant competence (D) priors: Beta(2,1) transformed to range [0.5, 1.0]
      - This prior slightly favors higher competence while allowing the full range of possible values
    - Consensus answer (Z) priors: Bernoulli(0.5)
      - This represents maximum uncertainty about the correct answers
    
    ## Convergence Assessment
    {convergence_status} was achieved with r_hat values {'all below 1.1' if converged else 'indicating some issues'}.
    
    ## Competence Estimates
    - Most competent informant: {top_informant['Informant_ID']} with {top_informant['Competence']:.3f} estimated competence
    - Least competent informant: {bottom_informant['Informant_ID']} with {bottom_informant['Competence']:.3f} estimated competence
    - Average informant competence: {competence_df['Competence'].mean():.3f}
    
    ## Consensus Answers vs Majority Vote
    - Number of questions: {len(consensus_df)}
    - Questions where CCT model disagrees with majority vote: {disagreements}
    
    ## Discussion
    Differences between the CCT model consensus and majority vote can be attributed to the CCT model's weighting of responses by informant competence. The model gives more weight to answers from informants with higher estimated competence, potentially leading to different consensus answers than simple majority voting when less competent informants form a majority. This demonstrates the key advantage of the CCT approach: it can identify the most likely correct answers even when they're not the most common responses, by accounting for differences in informant knowledge.
    """
    
    return report

def main():
    """
    Main function to run the CCT analysis.
    """
    # Set the filepath
    filepath = "../data/plant_knowledge.csv"
    
    # Load the data
    try:
        X, informant_ids = load_data(filepath)
        print(f"Loaded data with shape: {X.shape}")
    except FileNotFoundError:
        print(f"File not found: {filepath}")
        return
    
    # Build the CCT model
    cct_model = build_cct_model(X)
    
    # Run inference
    print("Running MCMC sampling...")
    trace = run_inference(cct_model)
    
    # Analyze results
    competence_df = analyze_competence(trace, informant_ids)
    consensus_df = analyze_consensus(trace, X)
    
    # Visualize results
    print("Generating visualizations...")
    visualize_results(trace, competence_df, consensus_df)
    
    # Generate report
    print("Generating report...")
    report = generate_report(trace, competence_df, consensus_df)
    
    # Save dataframes to CSV
    competence_df.to_csv('competence_results.csv', index=False)
    consensus_df.to_csv('consensus_results.csv', index=False)
    
    # Save report to markdown file
    with open('cct_report.md', 'w') as f:
        f.write(report)
    
    print("Analysis complete! Results saved to files.")
    
    return trace, competence_df, consensus_df

if __name__ == "__main__":
    main()