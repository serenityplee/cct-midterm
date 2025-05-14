#I used chatgpt for help thoughout the assignment
import numpy as np
import pandas as pd
import pymc as pm
import arviz as az
import matplotlib.pyplot as plt
import seaborn as sns

def load_data(filepath):
    """
    Load the plant knowledge dataset and return it as a numpy array.
    
    Parameters:
    -----------
    filepath : str
        Path to the CSV file containing the plant knowledge data
        
    Returns:
    --------
    np.ndarray
        2D array of binary responses where rows are informants and columns are questions
    list
        List of informant identifiers
    """
    # Load data from CSV
    data = pd.read_csv(filepath)
    
    # Extract informant IDs if they exist
    if 'Informant_ID' in data.columns:
        informant_ids = data['Informant_ID'].values.tolist()
        response_data = data.drop(columns=['Informant_ID']).values
    else:
        informant_ids = [f"Informant_{i+1}" for i in range(len(data))]
        response_data = data.values
    
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
        D_raw = pm.Beta('D_raw', alpha=2, beta=1, shape=N)
        D = pm.Deterministic('D', 0.5 + 0.5 * D_raw)  # Constrain between 0.5 and 1
        
        # Prior for consensus answers (Z)
        Z = pm.Bernoulli('Z', p=0.5, shape=M)
        
        # Calculate response probabilities
        p = Z * D[:, None] + (1 - Z) * (1 - D[:, None])
        
        # Likelihood
        pm.Bernoulli('X_obs', p=p, observed=X)
        
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
        trace = pm.sample(
            draws=draws,
            chains=chains,
            tune=tune,
            random_seed=42,
            target_accept=0.9,
            return_inferencedata=True
        )
    
    return trace

def analyze_competence(trace, informant_ids=None):
    """
    Analyze the competence estimates for each informant.
    
    Parameters:
    -----------
    trace : az.InferenceData
        MCMC samples from the model
    informant_ids : list, optional
        List of informant IDs
        
    Returns:
    --------
    pd.DataFrame
        DataFrame with competence estimates for each informant
    """
    # Get posterior samples for D
    post = az.extract(trace)
    mean_competence = post['D'].mean('sample').values
    
    # Handle informant IDs
    if informant_ids is None or len(informant_ids) != len(mean_competence):
        informant_ids = [f"Informant_{i+1}" for i in range(len(mean_competence))]
    
    return pd.DataFrame({
        'Informant_ID': informant_ids,
        'Competence': mean_competence
    }).sort_values('Competence', ascending=False)

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
    # Get posterior samples for Z
    post = az.extract(trace)
    mean_consensus = post['Z'].mean('sample').values
    
    # Calculate majority vote
    majority_vote = (X.mean(axis=0) > 0.5).astype(int)
    
    return pd.DataFrame({
        'Question': [f"Q{i+1}" for i in range(len(mean_consensus))],
        'Consensus_Probability': mean_consensus,
        'Consensus_Answer': (mean_consensus > 0.5).astype(int),
        'Majority_Vote': majority_vote,
        'Agreement': (mean_consensus > 0.5) == (majority_vote == 1)
    })

def visualize_results(trace, competence_df, consensus_df):
    """
    Visualize the results of the CCT analysis.
    """
    # Plot competence estimates
    plt.figure(figsize=(10, 6))
    sns.barplot(x='Informant_ID', y='Competence', data=competence_df)
    plt.title('Estimated Informant Competence')
    plt.ylabel('Competence')
    plt.xticks(rotation=45)
    plt.axhline(y=0.5, color='r', linestyle='--')
    plt.tight_layout()
    plt.savefig('competence_estimates.png')
    plt.close()
    
    # Plot consensus vs majority
    plt.figure(figsize=(12, 6))
    x = np.arange(len(consensus_df))
    width = 0.35
    
    plt.bar(x - width/2, consensus_df['Consensus_Probability'], width, label='CCT Probability')
    plt.bar(x + width/2, consensus_df['Majority_Vote'], width, label='Majority Vote', alpha=0.7)
    
    plt.xlabel('Question')
    plt.ylabel('Probability/Vote')
    plt.title('Consensus Answers vs Majority Vote')
    plt.xticks(x, consensus_df['Question'])
    plt.legend()
    plt.tight_layout()
    plt.savefig('consensus_vs_majority.png')
    plt.close()
    
    # Plot posterior distributions
    az.plot_posterior(trace, var_names=['D'])
    plt.tight_layout()
    plt.savefig('competence_posteriors.png')
    plt.close()

def generate_report(trace, competence_df, consensus_df):
    """
    Generate a report summarizing the CCT analysis results.
    """
    # Convergence diagnostics
    summary = az.summary(trace, var_names=['D', 'Z'])
    converged = (summary['r_hat'] < 1.05).all()
    
    # Top and bottom informants
    top = competence_df.iloc[0]
    bottom = competence_df.iloc[-1]
    
    # Disagreements
    disagreements = (~consensus_df['Agreement']).sum()
    
    report = f"""
    CCT Model Analysis Report
    
    Model Structure:
    - Competence (D): Beta(2,1) prior transformed to [0.5, 1]
    - Consensus (Z): Bernoulli(0.5) prior
    
    Convergence:
    - R-hat values all below 1.05: {'Yes' if converged else 'No'}
    
    Competence Estimates:
    - Highest: {top['Informant_ID']} ({top['Competence']:.3f})
    - Lowest: {bottom['Informant_ID']} ({bottom['Competence']:.3f})
    - Average: {competence_df['Competence'].mean():.3f}
    
    Consensus Results:
    - Total questions: {len(consensus_df)}
    - Questions where CCT ≠ Majority: {disagreements}
    """
    
    with open('cct_report.txt', 'w') as f:
        f.write(report)
    
    return report

def main():
    """Main analysis workflow"""
    try:
        # Load data
        X, informant_ids = load_data("../data/plant_knowledge.csv")
        print(f"Data loaded: {X.shape[0]} informants, {X.shape[1]} questions")
        
        # Build and run model
        model = build_cct_model(X)
        trace = run_inference(model)
        
        # Analyze results
        competence_df = analyze_competence(trace, informant_ids)
        consensus_df = analyze_consensus(trace, X)
        
        # Generate outputs
        visualize_results(trace, competence_df, consensus_df)
        report = generate_report(trace, competence_df, consensus_df)
        
        # Save results
        competence_df.to_csv('competence_results.csv', index=False)
        consensus_df.to_csv('consensus_results.csv', index=False)
        
        print("Analysis completed successfully")
        print(report)
        
        return trace, competence_df, consensus_df
        
    except Exception as e:
        print(f"Error in analysis: {str(e)}")
        raise

if __name__ == "__main__":
    main()