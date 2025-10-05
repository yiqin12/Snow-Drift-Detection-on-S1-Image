# If not installed:
# options(repos = c(CRAN = "https://cloud.r-project.org"))
# install.packages(c("readr","dplyr","lme4","performance"))
setwd("F:/Sd/final_SD")
library(readr)
library(dplyr)
library(lme4)
library(performance)  # r2_nakagawa

# 1) Read the original CSV file 
df <- read_csv("length_all_ROIselected.csv", show_col_types = FALSE)

# 2) Calculate trail_avg (without log transformation)
#    and construct hierarchical/standardized predictors (snowfall also not log-transformed)
df <- df %>%
  mutate(
    trail1_valid_int = as.integer(trail1_valid > 0),
    trail2_valid_int = as.integer(trail2_valid > 0),
    n_valid          = trail1_valid_int + trail2_valid_int
  ) %>%
  filter(n_valid > 0) %>%
  mutate(
    trail_avg = (trail1_length*trail1_valid_int + trail2_length*trail2_valid_int)/n_valid,
    slot_f    = factor(.data[["roi_slot"]]),
    day_f     = factor(day),
    wind_z    = as.numeric(scale(windspped_ms)),
    W_z       = as.numeric(scale(iceberg_width_m)),
    H_z       = as.numeric(scale(iceberg_height_m)),
    snow_z    = as.numeric(scale(pmax(snowfall_mum, 0)))  # Do not take log; standardize raw values directly
  )

# 3) LMM (raw scale of trail length, snowfall in raw scale; random structure same as before)
m_rawLen <- lmer(
  trail_avg ~ wind_z + snow_z + W_z + H_z +
    (1 + wind_z | slot_f) + (1 | day_f),
  data = df, REML = TRUE,
  control = lmerControl(optimizer = "bobyqa", optCtrl = list(maxfun = 1e5))
)

summary(m_rawLen)


# 3b) Test different random intercept structures
# Simple linear model (fixed effects only)
fm_fixed <- trail_avg ~ wind_z + snow_z + W_z + H_z
lm_raw   <- lm(fm_fixed, data = df)

cat("\n== OLS on raw scale (trail_avg) ==\n")
print(summary(lm_raw)$coefficients)  # Standard errors


# Baseline: random intercept only
m_int  <- lmer(trail_avg ~ wind_z + snow_z + W_z + H_z +
                 (1 | slot_f) + (1 | day_f),
               data=df, REML=FALSE)
summary(m_int)

# Add random slope for wind
m_wRS  <- lmer(trail_avg ~ wind_z + snow_z + W_z + H_z +
                 (1 + wind_z | slot_f) + (1 | day_f),
               data=df, REML=FALSE)
summary(m_wRS)

# Add random slope for snowfall
m_sRS  <- lmer(trail_avg ~ wind_z + snow_z + W_z + H_z +
                 (1 + snow_z | slot_f) + (1 | day_f),
               data=df, REML=FALSE)
summary(m_sRS)

# Add both
m_wsRS <- lmer(trail_avg ~ wind_z + snow_z + W_z + H_z +
                 (1 + wind_z + snow_z | slot_f) + (1 | day_f),
               data=df, REML=FALSE)
summary(m_wsRS)

# Model comparison & diagnostics断
anova(m_int, m_wRS, m_sRS, m_wsRS)   # AIC / LRT 表
isSingular(m_wRS); isSingular(m_sRS); isSingular(m_wsRS)
VarCorr(m_wRS)$slot_f                # Check whether wind_z random slope variance is significant
VarCorr(m_sRS)$slot_f                # Check whether snow_z random slope variance is close to zero




# 4) Diagnostics: residuals vs fitted (funnel shape often more visible on raw scale)
par(mfrow = c(1,1))
plot(fitted(m_rawLen), resid(m_rawLen),
     xlab = "Fitted", ylab = "Residuals",
     main = "LMM on raw length (no logs)")
abline(h = 0, lty = 2)

# 5) Model explanatory power (Nakagawa R²)
r2_nakagawa(m_rawLen)

# 6) If comparing fixed-effect models (e.g., with or without H),
#    use ML (maximum likelihood) version instead of REML
m_rawLen_ml <- update(m_rawLen, REML = FALSE)
AIC(m_rawLen_ml)
# anova(m_rawLen_ml, <another ML model object>)
