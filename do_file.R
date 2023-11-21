library(dplyr)
library(countrycode)
library(moments)
library(ordinalNet)
library(foreach)
library(doParallel)
library(here)
here()

jobfair_test <- read_csv(here("jobfair_test.csv"))
jobfair_train <- read_csv(here("jobfair_train.csv"))


# Train dataset -----------------------------------------------------------
summary(jobfair_train)
#NA's->global_competition_level; negative values->tokens_stash,rests_stash

# creating analysis dataset --------------------------------------------------------
# data cleaning
jobfair_train_ad_dc <- jobfair_train %>%
  mutate(
    global_competition_level = # global_competition_level -> NA's are considered as not having competition level. Hence it makes sense to atach zero to these
      case_when(is.na(global_competition_level) ~ 0,
                .default = global_competition_level),
    tokens_stash =
      case_when(tokens_stash < 0 ~ 0, # negative values of stashes
                .default = tokens_stash),
    rests_stash =
      case_when(rests_stash < 0 ~ 0, # negative values of stashes
                .default = rests_stash)
  )

# registration country is very big var which will consume huge amount of time for model estimation so grouping to continents
jobfair_train_ad_dc$continent <- countrycode(sourcevar =  jobfair_train %>% pull(registration_country),
                            origin = "country.name",
                            destination = "continent")
# Warning message:
#   Some values were not matched unambiguously: Anonymous Proxy, Europe, Kosovo, Unknown
jobfair_train_ad_dc$continent <- case_when(jobfair_train_ad_dc$registration_country=="Anonymous Proxy"~"Unknown",
                                           jobfair_train_ad_dc$registration_country=="Europe"~"Europe",
                                           jobfair_train_ad_dc$registration_country=="Kosovo"~"Europe",
                                           jobfair_train_ad_dc$registration_country=="Unknown"~"Unknown",
                                           .default = as.character(jobfair_train_ad_dc$continent)
                                           )
table(jobfair_train_ad_dc$continent,useNA = "ifany")
# converting to matrix format
# all categorical to dummy variables
jobfair_train_ad_ctmf <- model.matrix(~., jobfair_train_ad_dc %>% select(!registration_country))[,-1] 


# group-scaling -----------------------------------------------------------

#numeric variables
namesNum <- jobfair_train %>% select(where(is.numeric)) %>% names()
namesNum <- namesNum[!namesNum%in%c("season", "club_id", "league_id", "league_rank")]
#Dummy variables
namesDummy <- colnames(jobfair_train_ad_ctmf)[!colnames(jobfair_train_ad_ctmf)%in%
                         c(c("season", "club_id", "league_id", "league_rank"),namesNum)]


# teams attributes are valued are relative to middle team in the league 
# also different leagues have different dispersion/competition hence scaling needed
normaliseNum <- function(x){
  (x - mean(x))/(max(x)+0.00001)# since there are leagues where sd() is zero, max() safer option but need Inf for all-zero cases
}
normaliseDummy <- function(x){
  (x - mean(x))#mean is in this case equal to proportion. 
}#being a type of payer "Whale" is good, but if others in a league are "Whales" can offset that

jobfair_train_ad_gs <- as_tibble(jobfair_train_ad_ctmf) %>%
  group_by(league_id) %>% 
  mutate(across(all_of(namesNum), normaliseNum),
         across(all_of(namesDummy), normaliseDummy))


# modelling section -------------------------------------------------------
# because of (already missed) deadline :), I used random sample of 100 leagues (1400 teams)
set.seed(123)
sampling <- unique(jobfair_train$league_id)[sample(1:length(unique(jobfair_train$league_id)),size = 100)]

y_numeric <- jobfair_train_ad_gs %>% 
  filter(league_id%in%sampling) %>% 
  pull(league_rank)
y <- factor(y_numeric, ordered = TRUE)

x <- as.matrix(jobfair_train_ad_gs %>% ungroup() %>% 
                 filter(league_id%in%sampling) %>% 
                 select(!all_of(c("league_id","season", "club_id", "league_id", "league_rank"))))

n.cores <- parallel::detectCores() - 1
registerDoParallel(cores = n.cores)
seqAlpha <- c(0,0.33,0.66,1)#computational time consuming (even with parallelization) to loop through this  
# alpha elastic net coef - ridge <-> lasso
 
set.seed(1234)
system.time(  
  cross_valid <- foreach(i = seqAlpha, .combine = rbind, .packages="ordinalNet") %dopar% { 
fit_cv <- ordinalNetCV(x, y, family = "cumulative", link = "logit",alpha = i,
                   parallelTerms = TRUE, nonparallelTerms = FALSE, standardize = TRUE,
                   tuneMethod = "cvLoglik")#estimation used 20 lambda values
colMeans(summary(fit_cv))
  })

# best lambda -> based on cross-validation log-likelihood
bestLambda <- cross_valid[which.max(cross_valid[,2]),1]
bestAlpha <- seqAlpha[which.max(cross_valid[,2])]


#fit ordinal logistic regression with best cross-valid lambda and alpha
y_numeric <- jobfair_train_ad_gs %>% 
  pull(league_rank)
y <- factor(y_numeric, ordered = TRUE)

x <- as.matrix(jobfair_train_ad_gs %>% ungroup() %>% 
                 select(!all_of(c("league_id","season", "club_id", "league_id", "league_rank"))))

fit <- ordinalNet(x, y, family = "cumulative", link = "logit",alpha = bestAlpha,
                        parallelTerms = TRUE, nonparallelTerms = FALSE, 
                        standardize = TRUE, lambdaVals=bestLambda)

coef(fit)


# TEST dataset cleaning and prediction ----------------------------------------
summary(jobfair_test)
# data cleaning
jobfair_test_ad_dc <- jobfair_test %>%
  mutate(
    global_competition_level = # global_competition_level -> NA's are considered as not having competition level. Hence it makes sense to atach zero to these
      case_when(is.na(global_competition_level) ~ 0,
                .default = global_competition_level),
    tokens_stash =
      case_when(tokens_stash < 0 ~ 0, # negative values of stashes
                .default = tokens_stash),
    rests_stash =
      case_when(rests_stash < 0 ~ 0, # negative values of stashes
                .default = rests_stash)
  )

# registration country is very big var which will consume huge amount of time for model estimation so grouping to continents
jobfair_test_ad_dc$continent <- countrycode(sourcevar =  jobfair_test %>% pull(registration_country),
                                             origin = "country.name",
                                             destination = "continent")
# Warning message:
#   Some values were not matched unambiguously: Anonymous Proxy, Europe, Kosovo, Unknown
jobfair_test_ad_dc$continent <- case_when(jobfair_test_ad_dc$registration_country=="Anonymous Proxy"~"Unknown",
                                           jobfair_test_ad_dc$registration_country=="Europe"~"Europe",
                                           jobfair_test_ad_dc$registration_country=="Kosovo"~"Europe",
                                           jobfair_test_ad_dc$registration_country=="Unknown"~"Unknown",
                                           .default = as.character(jobfair_test_ad_dc$continent)
)
table(jobfair_test_ad_dc$continent,useNA = "ifany")

jobfair_test_ad_ctmf <- model.matrix(~., jobfair_test_ad_dc %>% select(!registration_country))[,-1] 


# group-scaling -----------------------------------------------------------
# group-scaling in order to (try to) eliminate hierarchical structure of the data. Alternative is to use hierarchical
# modelling approach, but not sure if it helps much compared to this approach

#numeric variables
namesNum <- jobfair_test %>% select(where(is.numeric)) %>% names()
namesNum <- namesNum[!namesNum%in%c("season", "club_id", "league_id", "league_rank")]
#Dummy variables
namesDummy <- colnames(jobfair_test_ad_ctmf)[!colnames(jobfair_test_ad_ctmf)%in%
                                                c(c("season", "club_id", "league_id", "league_rank"),namesNum)]


# teams attributes are valued are relative to middle team in the league 
# also different leagues have different dispersion/competition hence scaling needed
normaliseNum <- function(x){
  (x - mean(x))/(max(x)+0.00001)# since there are leagues where sd() is zero, max() safer option but need Inf for all-zero cases
}
normaliseDummy <- function(x){
  (x - mean(x))#mean is in this case equal to proportion. 
}#being a type of payer "Whale" is good, but if others in a league are "Whales" can offset that

jobfair_test_ad_gs <- as_tibble(jobfair_test_ad_ctmf) %>%
  group_by(league_id) %>% 
  mutate(across(all_of(namesNum), normaliseNum),
         across(all_of(namesDummy), normaliseDummy))

summary(jobfair_test_ad_gs)

identical(colnames(jobfair_test_ad_gs),
colnames(jobfair_train_ad_gs)[!grepl("league_rank", colnames(jobfair_train_ad_gs))])

# modelling section -------------------------------------------------------
x_test <- as.matrix(jobfair_test_ad_gs %>% ungroup() %>%
                 select(!all_of(c("league_id","season", "club_id", "league_id"))))

identical(colnames(x_test),colnames(x))
# Prediction and submission set -------------------------------------------
Utility_test <- x_test%*%coef(fit)[-(1:13)] # coefs from the Train set


league_rank_predictions <-  jobfair_test %>% 
  mutate(Utility_test=Utility_test) %>%
  group_by(league_id) %>%
  mutate(league_rank = 15-rank(Utility_test, ties.method = "random")) %>% 
  ungroup() %>% 
  select(club_id, league_rank)  

write.csv(league_rank_predictions,here("league_rank_predictions.csv"),row.names = FALSE)
