# At the top of retrofit_calc.py
from .logging_config import get_logger

logger = get_logger(__name__)

import pandas as pd 
import os 
from src.retrofit_calc import process_postcodes_for_retrofit_with_uncertainty


def process_retrofit_batch_with_uncertainty(
    pc_batch, 
    data, 
    INPUT_GPK, 
    process_batch_name, 
    log_file, 
    region,
    retrofit_config,
    retrofig_model,
    scenarios,
    energy_column='total_gas',
    n_monte_carlo=100,
    random_seed=42,
    use_uncertainty=True
):
    """
    Process batch of postcodes with optional uncertainty analysis.
    
    Args:
        pc_batch: List of postcodes to process
        data: ONSUD data
        INPUT_GPK: GPK input file
        process_batch_name: Name for logging
        log_file: Path to save results
        region: Region code (LN, SE, etc.)
        energy_column: Column name with energy consumption data
        n_monte_carlo: Number of Monte Carlo iterations (default 10000)
        random_seed: Random seed for reproducibility
        use_uncertainty: If True, runs Monte Carlo; if False, uses legacy cost-only method
    """
    logger.debug('Starting batch processing with uncertainty analysis...')
    
    results = []
    failed_pcs = []
    
    for pc in pc_batch:
        logger.debug(f'Processing postcode: {pc}')
        
        try:
            if use_uncertainty:
                pc_result = process_postcodes_for_retrofit_with_uncertainty(
                    pc=pc,
                    onsud_data=data,
                    INPUT_GPK=INPUT_GPK,
                    region=region,
                    retrofit_config=retrofit_config, 
                    retrofig_model=retrofig_model,
                    scenarios=scenarios,

                    energy_column=energy_column,
                    n_monte_carlo=n_monte_carlo,
                    random_seed=random_seed
                )
   
            
                if pc_result is not None:
                    # Check for errors in result
                    if 'error' in pc_result and pc_result['error'] is not None:
                        logger.warning(f"Postcode {pc} failed: {pc_result['error']}")
                        failed_pcs.append({'postcode': pc, 'error': pc_result['error']})
                    else:
                        results.append(pc_result)
                else:
                    logger.warning(f"Postcode {pc} returned None")
                    failed_pcs.append({'postcode': pc, 'error': 'Returned None'})
                
        except Exception as e:
            logger.error(f"Exception processing postcode {pc}: {str(e)}")
            failed_pcs.append({'postcode': pc, 'error': str(e)})
    
    logger.info(f'Processed: {len(results)} successful, {len(failed_pcs)} failed')
    
    # Save successful results
    if results:
        
        logger.info(f'Saving {len(results)} results')
    
        # DIAGNOSTIC: Check what we're working with
        logger.debug(f"Number of results to concatenate: {len(results)}")
        logger.debug(f"Type of first result: {type(results[0])}")
        
        if isinstance(results[0], pd.DataFrame):
            logger.debug(f"First result columns: {results[0].columns.tolist()}")
            logger.debug(f"First result shape: {results[0].shape}")
            logger.debug(f"First result head:\n{results[0].head()}")
        elif isinstance(results[0], dict):
            logger.debug(f"First result keys: {results[0].keys()}")
            logger.debug(f"First result: {results[0]}")
        else:
            logger.error(f"Unexpected result type: {type(results[0])}")
            logger.error(f"First result: {results[0]}")
        
        try:
            df = pd.concat(results, ignore_index=True)
            logger.info(f"Concatenated DataFrame shape: {df.shape}")
            logger.info(f"Concatenated DataFrame columns: {df.columns.tolist()}")
            # Check if 'postcode' column exists
            if 'postcode' not in df.columns:
                logger.error(f"CRITICAL ERROR: 'postcode' column missing!")
                logger.error(f"Available columns: {df.columns.tolist()}")
                logger.error(f"Sample row:\n{df.iloc[0]}")
                raise KeyError("Required 'postcode' column not found in results")
            
            
            # Save results
            if not os.path.exists(log_file):
                logger.info(f'Creating new log file: {log_file}')
                df.to_csv(log_file, index=False)
            else:
                logger.info(f'Appending to existing log file: {log_file}')
                df.to_csv(log_file, mode='a', header=False, index=False)
            
            logger.info(f'Successfully saved {len(results)} results for batch: {process_batch_name}')
        
        except KeyError as ke:
            logger.error(f"KeyError during save: {str(ke)}")
            logger.error(f"This usually means the DataFrame structure is incorrect")
            raise
        except Exception as e:
            logger.error(f"Unexpected error during save: {type(e).__name__}: {str(e)}")
            raise
        
        
       
    if failed_pcs:
        failed_log = log_file.replace('.csv', '_failed.csv')
        df_failed = pd.DataFrame(failed_pcs)
        
        if not os.path.exists(failed_log):
            df_failed.to_csv(failed_log, index=False)
        else:
            df_failed.to_csv(failed_log, mode='a', header=False, index=False)
        
        logger.warning(f'Saved {len(failed_pcs)} failed postcodes to: {failed_log}')


def run_retrofit_calc_with_uncertainty(
    pcs_list, 
    data, 
    INPUT_GPK, 
    batch_size, 
    batch_label, 
    log_file,
    retrofit_config, 
    retrofig_model,
    scenarios,
    energy_column='total_gas_derived',
    n_monte_carlo=100,
    random_seed=42,
    use_uncertainty=True,
 
):
    """
    Run retrofit calculations with uncertainty analysis for list of postcodes.
    calls process_retrofit_batch_with_uncertainty to proces batch 
    
    Args:
        pcs_list: List of postcodes to process
        data: ONSUD data
        INPUT_GPK: GPK input file
        batch_size: Number of postcodes per batch
        batch_label: Label for logging
        log_file: Path to save results (region extracted from path)
        energy_column: Column name with energy consumption
        n_monte_carlo: Number of Monte Carlo iterations
        random_seed: Random seed for reproducibility
        use_uncertainty: If True, runs Monte Carlo; if False, legacy cost-only
    
    Example:
        run_retrofit_calc_with_uncertainty(
            pcs_list=postcodes,
            data=onsud_data,
            INPUT_GPK=gpk_file,
            batch_size=100,
            batch_label='london_batch_1',
            log_file='outputs/LN/retrofit_results.csv',
            energy_column='total_gas',
            n_monte_carlo=10000,
            use_uncertainty=True
        )
    """
    # Extract region from file path
    region = log_file.split('/')[-2]
    logger.info(f'Extracted region: {region}')
    
    # Validate region
    valid_regions = ['NW', 'NE', 'LN', 'WA', 'WM', 'EM', 'EE', 'SE', 'SW', 'YH']
    if region not in valid_regions:
        raise ValueError(f'Invalid region "{region}". Valid regions: {valid_regions}')
    
    # Log configuration
    logger.info(f'Processing {len(pcs_list)} postcodes in region {region}')
    logger.info(f'Logger_Batch size: {batch_size}')
    logger.info(f'Uncertainty analysis: {"ENABLED" if use_uncertainty else "DISABLED"}')
    if use_uncertainty:
        logger.info(f'Monte Carlo iterations: {n_monte_carlo}')
        logger.info(f'Energy column: {energy_column}')
    
    # Process in batches
    total_batches = (len(pcs_list) + batch_size - 1) // batch_size
    
    for batch_num, i in enumerate(range(0, len(pcs_list), batch_size), 1):
        batch = pcs_list[i:i+batch_size]
         
        logger.info(f'Processing batch {batch_num}/{total_batches} ({len(batch)} postcodes)')
        
        try:
            process_retrofit_batch_with_uncertainty(
                pc_batch=batch,
                data=data,
             
                INPUT_GPK=INPUT_GPK,
                process_batch_name=f'{batch_label}_batch_{batch_num}',
                log_file=log_file,
                retrofit_config= retrofit_config,
                retrofig_model=retrofig_model,
                scenarios=scenarios,
                region=region,
                energy_column=energy_column,
                n_monte_carlo=n_monte_carlo,
                random_seed=random_seed,  
                use_uncertainty=use_uncertainty
            )
            logger.info(f'Completed batch {batch_num}/{total_batches}')
            
        except Exception as e:
            logger.error(f'Failed batch {batch_num}/{total_batches}: {str(e)}')
            # Continue to next batch rather than failing entire job
            continue
    
    logger.info(f'Completed all batches for {batch_label}')


# # Legacy function - kept for backwards compatibility
# def process_retrofit_batch(pc_batch, data, INPUT_GPK, process_batch_name, log_file, region):
#     """Legacy function - costs only, no uncertainty analysis."""
#     return process_retrofit_batch_with_uncertainty(
#         pc_batch=pc_batch,
#         data=data,
#         INPUT_GPK=INPUT_GPK,
#         process_batch_name=process_batch_name,
#         log_file=log_file,
#         region=region,
#         use_uncertainty=False
#     )


# def run_retrofit_calc(pcs_list, data, INPUT_GPK, batch_size, batch_label, log_file):
#     """Legacy function - costs only, no uncertainty analysis."""
#     return run_retrofit_calc_with_uncertainty(
#         pcs_list=pcs_list,
#         data=data,
#         INPUT_GPK=INPUT_GPK,
#         batch_size=batch_size,
#         batch_label=batch_label,
#         log_file=log_file,
#         use_uncertainty=False
#     )