import json
import pickle
import sqlite3 as db

################################################################################
## The Saver class provides an easy way to save or load, an agent, its network
#  and statistics about its performances
################################################################################
class Saver:

    ## Type for an agent implementing the algorithm developped by deepmind
    #  in their paper of 2013
    DEEP_MIND_AGENT = "DeepMindAgent"

    ## The Saver class constructor initialize the Saver object. 
    #
    #  The constructor connects to or creates the sqlite database to use and
    #  creates the tables it needs if its required.
    #
    #   @param dbPath : The path to the database to connect to or to create
    def __init__(self, dbPath):
        self._conn = db.connect(dbPath, check_same_thread = False)

        c = self._conn.cursor()

        c.execute("""CREATE TABLE IF NOT EXISTS
                     agents (id     INTEGER PRIMARY KEY AUTOINCREMENT,
                             name   TEXT,
                             type   TEXT,
                             ts     DATETIME DEFAULT CURRENT_TIMESTAMP,
                             params TEXT)""")
        
        c.execute("""CREATE TABLE IF NOT EXISTS
                     networks (id       INTEGER PRIMARY KEY AUTOINCREMENT,
                               id_agent INTEGER,
                               info     TEXT,
                               ts       DATETIME DEFAULT CURRENT_TIMESTAMP,
                               network  BLOB)""")

        c.execute("""CREATE TABLE IF NOT EXISTS
                     datasets (id    INTEGER PRIMARY KEY AUTOINCREMENT,
                               size  INTEGER,
                               shape TEXT,
                               data  BLOB)""")
        
        c.execute("""CREATE TABLE IF NOT EXISTS
                     stats (id         INTEGER PRIMARY KEY AUTOINCREMENT,
                            id_agent   INTEGER,
                            id_network INTEGER,
                            name       TEXT,
                            epoch      INTEGER,
                            value      REAL)""")
        
        self._conn.commit()


    ## The listAgents query the database and returns a list of tuples
    #  representing the available agents
    #
    #   @return A list of tuples with agent's attributes : (id, name, type,
    #           creation time stamp, parameters)
    def listAgents(self):
        c   = self._conn.cursor()
        res = c.execute("SELECT * FROM agents").fetchall()
        return res

    ## The listNetworks method returns a list all the network availables for the
    #  given agent
    #
    #   @param agentId : The id of the agent
    #
    #   @return A list of tuples with the parameters of the available networks
    #           (id, agent's id, informations, creation timestamp, network)
    def listNetworks(self, agentId):
        c   = self._conn.cursor()
        res = c.execute("SELECT * FROM networks WHERE id_agent = ?",
                        (agentId,)).fetchall()
        return res

    ## The listDatasets method returns a list of the datasets matching the given
    #  shape and which length is between the given sizes
    #
    #   @param shape   : The shape of the desired data set
    #   @param minSize : The minimum number of elements in the dataset
    #   @param maxSize : The maximum number of elements in the dataset
    #
    #   @return A list of tuples containing the id and the size of the available
    #           datasets (id, size)
    def listDatasets(self, shape, minSize, maxSize):
        c   = self._conn.cursor()
        res = c.execute("""SELECT id,size
                           FROM   datasets
                           WHERE  shape = ?
                             AND  size >= ?
                             AND  size <= ?""",
                        (json.dumps(shape), minSize,maxSize)).fetchall()
        return res

    ## The newAgent method records a new agent in the database
    #
    #   @param name      : The name of the agent
    #   @param agentType : The agent's type
    #   @param params    : The agent's parameters to save
    #
    #   @return The id of the newly created agent
    def newAgent(self, name, agentType, params):
        c = self._conn.cursor()
        c.execute("INSERT INTO agents (name, type, params) VALUES (?,?,?)",
                  (name, agentType, json.dumps(params)))

        agentId = c.execute("SELECT id FROM agents WHERE ROWID = ?",
                            (c.lastrowid,)).fetchone()[0]
        self._conn.commit()

        return agentId

    ## The saveAgent methods overwrites the parameters recorder for the given
    #  agent.
    #
    #   @param agentId : The id of the agent to update
    #   @param params  : The new parameters to save
    def saveAgent(self, agentId, params):
        c = self._conn.cursor()
        c.execute("UPDATE agents SET params = ? WHERE id = ?",
                  (json.dumps(params), agentId))
        self._conn.commit()

    ## The saveNetwork method adds the given network to the list of the networks
    # available for the given agent
    #
    #   @param agentId : The id of the agent the network to save belongs to
    #   @param info    : Some information associated to network to save
    #   @param network : The network to save
    #
    #   @return The id of the newly saved network
    def saveNetwork(self, agentId, info, network):
        c = self._conn.cursor()
        c.execute("""INSERT INTO networks (id_agent, info, network)
                     VALUES (?,?,?)""", (agentId, info, pickle.dumps(network)))
        
        networkId = c.execute("SELECT id FROM networks WHERE ROWID = ?",
                              (c.lastrowid,)).fetchone()[0]
        self._conn.commit()
        return networkId
        
    ## The saveStat method saves the given statistics into the database
    #
    #   @param agentId   : The id of the agent assiciated to the statistics to
    #                      save
    #   @param networkId : The id of the network used to perform these
    #                      statistics
    #   @param name      : A name associated to these statistics
    #   @param epoch     : The epoch
    #   @param value     : The value to save
    def saveStat(self, agentId, networkId, name, epoch, value):
        c = self._conn.cursor()
        c.execute("""INSERT INTO
                     stats  (id_agent, id_network, name, epoch, value)
                     VALUES (?,?,?,?,?)""", 
                     (agentId, networkId, name, epoch, value))
        self._conn.commit()

    ## The newDataset method save a new dataset into the database
    #
    #   @param length : The number of elemnts in the dataset
    #   @param shape  : The shape of the dataset to save
    #   @param data   : The dataset to save
    #
    #   @return The id of the newly created dataset
    def newDataset(self, length, shape, data):
        c = self._conn.cursor()
        c.execute("INSERT INTO datasets (size, shape, data) VALUES (?,?,?)",
                  (length, json.dumps(shape), pickle.dumps(data)))
        setId = c.execute("SELECT id FROM datasets WHERE ROWID = ?",
                          (c.lastrowid,)).fetchone()[0]
        self._conn.commit()
        return setId

    ## The loadAgent method returns the parameters associated to the given agent
    #
    #   @param agentId : The id of the desired agent
    #
    #   @return The previously saved parameters
    def loadAgent(self, agentId):
        c = self._conn.cursor()
        p = c.execute("""SELECT params
                         FROM   agents
                         WHERE  agents.id = ?""", (agentId,)).fetchone()[0]
        return json.loads(p)

    ## The loadNetwork method returns the desired network
    #
    #   @param agentId   : The id of the agent that created the network
    #   @param networkId : The id of the network to load. If None (default), the
    #                      last saved network is returned
    #
    #   @return The previously saved network
    def loadNetwork(self, agentId, networkId = None):
        c = self._conn.cursor()
        if networkId is None :
            n = c.execute("""SELECT   network
                             FROM     networks
                             WHERE    networks.id_agent = ?
                             ORDER BY networks.ts DESC
                             LIMIT    1""", (agentId,)).fetchone()[0]
        else:
            n = c.execute("""SELECT network
                             FROM   networks
                             WHERE  networks.id_agent = ? AND
                                    networks.id       = ?""",
                          (agentId, networkId)).fetchone()[0]
        return pickle.loads(n)

    ## The loadDataset method return the desired dataset
    #
    #   @param setId : The id of the dataset to return
    #
    #   @return The dataset previously saved
    def loadDataset(self, setId):
        c = self._conn.cursor()
        dataset = c.execute("SELECT data FROM datasets WHERE id = ?",
                            (setId,)).fetchone()[0]
        return pickle.loads(dataset)

